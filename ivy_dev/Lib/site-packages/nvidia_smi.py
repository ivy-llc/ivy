#####
# Copyright (c) 2011-2015, NVIDIA Corporation.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the NVIDIA Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
# THE POSSIBILITY OF SUCH DAMAGE.
#####

#
# nvidia_smi
# nvml_bindings <at> nvidia <dot> com
#
# Sample code that attempts to reproduce the output of nvidia-smi -q -x
# For many cases the output should match
#
# Can be used as a library or a command line script
#
# To Run:
# $ python nvidia_smi.py
#

from pynvml import *
import datetime

#
# Helper functions
#
def GetEccByType(handle, counterType, errorType):
    strResult = ''
    
    try:
        deviceMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                       NVML_MEMORY_LOCATION_DEVICE_MEMORY)
    except NVMLError as err:
        deviceMemory = handleError(err)
    strResult += '          <device_memory>' + str(deviceMemory) + '</device_memory>\n'
    
    try:
        registerFile = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                       NVML_MEMORY_LOCATION_REGISTER_FILE)
    except NVMLError as err:
        registerFile = handleError(err)
    
    strResult += '          <register_file>' + str(registerFile) + '</register_file>\n'
    
    try:
        l1Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                  NVML_MEMORY_LOCATION_L1_CACHE)
    except NVMLError as err:
        l1Cache = handleError(err)
    strResult += '          <l1_cache>' + str(l1Cache) + '</l1_cache>\n'
    
    try:
        l2Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                  NVML_MEMORY_LOCATION_L2_CACHE)
    except NVMLError as err:
        l2Cache = handleError(err)
    strResult += '          <l2_cache>' + str(l2Cache) + '</l2_cache>\n'
    
    try:
        textureMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                        NVML_MEMORY_LOCATION_TEXTURE_MEMORY)
    except NVMLError as err:
        textureMemory = handleError(err)
    strResult += '          <texture_memory>' + str(textureMemory) + '</texture_memory>\n'
    
    try:
        count = str(nvmlDeviceGetTotalEccErrors(handle, errorType, counterType))
    except NVMLError as err:
        count = handleError(err)
    strResult += '          <total>' + count + '</total>\n'
    
    return strResult

def GetEccByCounter(handle, counterType):
    strResult = ''
    strResult += '        <single_bit>\n'
    strResult += str(GetEccByType(handle, counterType, NVML_MEMORY_ERROR_TYPE_CORRECTED))
    strResult += '        </single_bit>\n'
    strResult += '        <double_bit>\n'
    strResult += str(GetEccByType(handle, counterType, NVML_MEMORY_ERROR_TYPE_UNCORRECTED))
    strResult += '        </double_bit>\n'
    return strResult

def GetEccStr(handle):
    strResult = ''
    strResult += '      <volatile>\n'
    strResult += str(GetEccByCounter(handle, NVML_VOLATILE_ECC))
    strResult += '      </volatile>\n'
    strResult += '      <aggregate>\n'
    strResult += str(GetEccByCounter(handle, NVML_AGGREGATE_ECC))
    strResult += '      </aggregate>\n'
    return strResult

def GetRetiredPagesByCause(handle, cause):
    strResult = ''
    try:
        pages = nvmlDeviceGetRetiredPages(handle, cause)   
        count = str(len(pages))
    except NVMLError as err:
        error = handleError(err)
        pages = None
        count = error
    strResult += '        <retired_count>' + count + '</retired_count>\n'
    if pages is not None:
        strResult += '        <retired_page_addresses>\n'
        for page in pages:
            strResult += '          <retired_page_address>' + "0x%016x" % page + '</retired_page_address>\n'
        strResult += '        </retired_page_addresses>\n'
    else:
        strResult += '        <retired_page_addresses>' + error + '</retired_page_addresses>\n'
    return strResult

def GetRetiredPagesStr(handle):
    strResult = ''
    causes = [ "multiple_single_bit_retirement", "double_bit_retirement" ]
    for idx in range(NVML_PAGE_RETIREMENT_CAUSE_COUNT):
        strResult += '      <' + causes[idx] + '>\n'
        strResult += GetRetiredPagesByCause(handle, idx)
        strResult += '      </' + causes[idx] + '>\n'

    strResult += '      <pending_retirement>'
    try:
        if NVML_FEATURE_DISABLED == nvmlDeviceGetRetiredPagesPendingStatus(handle):
            strResult += "No"
        else:
            strResult += "Yes"
    except NVMLError as err:
        strResult += handleError(err)
    strResult += '</pending_retirement>\n'
    return strResult
    
def StrGOM(mode):
    if mode == NVML_GOM_ALL_ON:
        return "All On";
    elif mode == NVML_GOM_COMPUTE:
        return "Compute";
    elif mode == NVML_GOM_LOW_DP:
        return "Low Double Precision";
    else:
        return "Unknown";

def GetClocksThrottleReasons(handle):
    throttleReasons = [
            [nvmlClocksThrottleReasonGpuIdle,           "clocks_throttle_reason_gpu_idle"],
            [nvmlClocksThrottleReasonUserDefinedClocks, "clocks_throttle_reason_user_defined_clocks"],
            [nvmlClocksThrottleReasonApplicationsClocksSetting, "clocks_throttle_reason_applications_clocks_setting"],
            [nvmlClocksThrottleReasonSwPowerCap,        "clocks_throttle_reason_sw_power_cap"],
            [nvmlClocksThrottleReasonHwSlowdown,        "clocks_throttle_reason_hw_slowdown"],
            [nvmlClocksThrottleReasonUnknown,           "clocks_throttle_reason_unknown"]
            ];

    strResult = ''

    try:
        supportedClocksThrottleReasons = nvmlDeviceGetSupportedClocksThrottleReasons(handle);
        clocksThrottleReasons = nvmlDeviceGetCurrentClocksThrottleReasons(handle);
        strResult += '    <clocks_throttle_reasons>\n'
        for (mask, name) in throttleReasons:
            if (name != "clocks_throttle_reason_user_defined_clocks"):
                if (mask & supportedClocksThrottleReasons):
                    val = "Active" if mask & clocksThrottleReasons else "Not Active";
                else:
                    val = handleError(NVML_ERROR_NOT_SUPPORTED);
                strResult += "      <%s>%s</%s>\n" % (name, val, name);
        strResult += '    </clocks_throttle_reasons>\n'
    except NVMLError as err:
        strResult += '    <clocks_throttle_reasons>%s</clocks_throttle_reasons>\n' % (handleError(err));

    return strResult;
        
#
# Converts errors into string messages
#
def handleError(err):
    if (err.value == NVML_ERROR_NOT_SUPPORTED):
        return "N/A"
    else:
        return err.__str__()

#######
def XmlDeviceQuery():

    strResult = ''
    try:
        #
        # Initialize NVML
        #
        nvmlInit()

        strResult += '<?xml version="1.0" ?>\n'
        strResult += '<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v4.dtd">\n'
        strResult += '<nvidia_smi_log>\n'

        strResult += '  <timestamp>' + str(datetime.date.today()) + '</timestamp>\n'
        strResult += '  <driver_version>' + str(nvmlSystemGetDriverVersion()) + '</driver_version>\n'

        deviceCount = nvmlDeviceGetCount()
        strResult += '  <attached_gpus>' + str(deviceCount) + '</attached_gpus>\n'

        for i in range(0, deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            
            pciInfo = nvmlDeviceGetPciInfo(handle)    
            
            strResult += '  <gpu id="%s">\n' % pciInfo.busId
            
            strResult += '    <product_name>' + nvmlDeviceGetName(handle) + '</product_name>\n'
            
            brandNames = {NVML_BRAND_UNKNOWN :  "Unknown",
                          NVML_BRAND_QUADRO  :  "Quadro",
                          NVML_BRAND_TESLA   :  "Tesla",
                          NVML_BRAND_NVS     :  "NVS",
                          NVML_BRAND_GRID    :  "Grid",
                          NVML_BRAND_GEFORCE :  "GeForce",
            }

            try:
                # if nvmlDeviceGetBrand() succeeds it is guaranteed to be in the dictionary
                brandName = brandNames[nvmlDeviceGetBrand(handle)]
            except NVMLError as err:
                brandName = handleError(err)


            strResult += '    <product_brand>' + brandName + '</product_brand>\n'
                
            try:
                state = ('Enabled' if (nvmlDeviceGetDisplayMode(handle) != 0) else 'Disabled')
            except NVMLError as err:
                state = handleError(err)
            
            strResult += '    <display_mode>' + state + '</display_mode>\n'
            
            try:
                state = ('Enabled' if (nvmlDeviceGetDisplayActive(handle) != 0) else 'Disabled')
            except NVMLError as err:
                state = handleError(err)
            
            strResult += '    <display_active>' + state + '</display_active>\n'
            
            try:
                mode = 'Enabled' if (nvmlDeviceGetPersistenceMode(handle) != 0) else 'Disabled'
            except NVMLError as err:
                mode = handleError(err)
            
            strResult += '    <persistence_mode>' + mode + '</persistence_mode>\n'
            
            try:
                mode = 'Enabled' if (nvmlDeviceGetAccountingMode(handle) != 0) else 'Disabled'
            except NVMLError as err:
                mode = handleError(err)
            
            strResult += '    <accounting_mode>' + mode + '</accounting_mode>\n'
           
            try:
                bufferSize = str(nvmlDeviceGetAccountingBufferSize(handle))
            except NVMLError as err:
                bufferSize = handleError(err)
            
            strResult += '    <accounting_mode_buffer_size>' + bufferSize + '</accounting_mode_buffer_size>\n'
                
            strResult += '    <driver_model>\n'

            try:
                current = 'WDDM' if (nvmlDeviceGetCurrentDriverModel(handle) == NVML_DRIVER_WDDM) else 'TCC' 
            except NVMLError as err:
                current = handleError(err)
            strResult += '      <current_dm>' + current + '</current_dm>\n'

            try:
                pending = 'WDDM' if (nvmlDeviceGetPendingDriverModel(handle) == NVML_DRIVER_WDDM) else 'TCC' 
            except NVMLError as err:
                pending = handleError(err)

            strResult += '      <pending_dm>' + pending + '</pending_dm>\n'

            strResult += '    </driver_model>\n'

            try:
                serial = nvmlDeviceGetSerial(handle)
            except NVMLError as err:
                serial = handleError(err)

            strResult += '    <serial>' + serial + '</serial>\n'

            try:
                uuid = nvmlDeviceGetUUID(handle)
            except NVMLError as err:
                uuid = handleError(err)

            strResult += '    <uuid>' + uuid + '</uuid>\n'
            
            try:
                minor_number = nvmlDeviceGetMinorNumber(handle)
            except NVMLError as err:
                minor_number = handleError(err)

            strResult += '    <minor_number>' + str(minor_number) + '</minor_number>\n'
            
            try:
                vbios = nvmlDeviceGetVbiosVersion(handle)
            except NVMLError as err:
                vbios = handleError(err)

            strResult += '    <vbios_version>' + vbios + '</vbios_version>\n'

            try:
                multiGpuBool = nvmlDeviceGetMultiGpuBoard(handle)
            except NVMLError as err:
                multiGpuBool = handleError(err);

            if multiGpuBool == "N/A":
                strResult += '    <multigpu_board>' + 'N/A' + '</multigpu_board>\n'
            elif multiGpuBool:
                strResult += '    <multigpu_board>' + 'Yes' + '</multigpu_board>\n'
            else:
                strResult += '    <multigpu_board>' + 'No' + '</multigpu_board>\n'

            try:
                boardId = nvmlDeviceGetBoardId(handle)
            except NVMLError as err:
                boardId = handleError(err)

            try:
                hexBID = "0x%x" % boardId
            except: 
                hexBID = boardId

            strResult += '    <board_id>' + hexBID + '</board_id>\n'

            strResult += '    <inforom_version>\n'
            
            try:
                img = nvmlDeviceGetInforomImageVersion(handle)
            except NVMLError as err:
                img = handleError(err)
                
            strResult += '      <img_version>' + img + '</img_version>\n'

            try:
                oem = nvmlDeviceGetInforomVersion(handle, NVML_INFOROM_OEM)
            except NVMLError as err:
                oem = handleError(err)
                
            strResult += '      <oem_object>' + oem + '</oem_object>\n'
            
            try:
                ecc = nvmlDeviceGetInforomVersion(handle, NVML_INFOROM_ECC)
            except NVMLError as err:
                ecc = handleError(err)
            
            strResult += '      <ecc_object>' + ecc + '</ecc_object>\n'

            try:
                pwr = nvmlDeviceGetInforomVersion(handle, NVML_INFOROM_POWER)
            except NVMLError as err:
                pwr = handleError(err)
            
            strResult += '      <pwr_object>' + pwr + '</pwr_object>\n'
                       
            strResult += '    </inforom_version>\n'

            strResult += '    <gpu_operation_mode>\n'

            try:
                current = StrGOM(nvmlDeviceGetCurrentGpuOperationMode(handle))
            except NVMLError as err:
                current = handleError(err)
            strResult += '      <current_gom>' + current + '</current_gom>\n'

            try:
                pending = StrGOM(nvmlDeviceGetPendingGpuOperationMode(handle))
            except NVMLError as err:
                pending = handleError(err)

            strResult += '      <pending_gom>' + pending + '</pending_gom>\n'

            strResult += '    </gpu_operation_mode>\n'

            strResult += '    <pci>\n'
            strResult += '      <pci_bus>%02X</pci_bus>\n' % pciInfo.bus
            strResult += '      <pci_device>%02X</pci_device>\n' % pciInfo.device
            strResult += '      <pci_domain>%04X</pci_domain>\n' % pciInfo.domain
            strResult += '      <pci_device_id>%08X</pci_device_id>\n' % (pciInfo.pciDeviceId)
            strResult += '      <pci_bus_id>' + str(pciInfo.busId) + '</pci_bus_id>\n'
            strResult += '      <pci_sub_system_id>%08X</pci_sub_system_id>\n' % (pciInfo.pciSubSystemId)
            strResult += '      <pci_gpu_link_info>\n'


            strResult += '        <pcie_gen>\n'

            try:
                gen = str(nvmlDeviceGetMaxPcieLinkGeneration(handle))
            except NVMLError as err:
                gen = handleError(err)

            strResult += '          <max_link_gen>' + gen + '</max_link_gen>\n'

            try:
                gen = str(nvmlDeviceGetCurrPcieLinkGeneration(handle))
            except NVMLError as err:
                gen = handleError(err)

            strResult += '          <current_link_gen>' + gen + '</current_link_gen>\n'
            strResult += '        </pcie_gen>\n'
            strResult += '        <link_widths>\n'

            try:
                width = str(nvmlDeviceGetMaxPcieLinkWidth(handle)) + 'x'
            except NVMLError as err:
                width = handleError(err)

            strResult += '          <max_link_width>' + width + '</max_link_width>\n'

            try:
                width = str(nvmlDeviceGetCurrPcieLinkWidth(handle)) + 'x'
            except NVMLError as err:
                width = handleError(err)

            strResult += '          <current_link_width>' + width + '</current_link_width>\n'

            strResult += '        </link_widths>\n'
            strResult += '      </pci_gpu_link_info>\n'


            strResult += '      <pci_bridge_chip>\n'

            try:
                bridgeHierarchy = nvmlDeviceGetBridgeChipInfo(handle)
                bridge_type = ''
                if bridgeHierarchy.bridgeChipInfo[0].type == 0:
                    bridge_type += 'PLX'
                else:
                    bridge_type += 'BR04'                    
                strResult += '        <bridge_chip_type>' + bridge_type + '</bridge_chip_type>\n'

                if bridgeHierarchy.bridgeChipInfo[0].fwVersion == 0:
                    strFwVersion = 'N/A'
                else:
                    strFwVersion = '%08X' % (bridgeHierarchy.bridgeChipInfo[0].fwVersion)
                strResult += '        <bridge_chip_fw>%s</bridge_chip_fw>\n' % (strFwVersion)
            except NVMLError as err:
                strResult += '        <bridge_chip_type>' + handleError(err) + '</bridge_chip_type>\n'
                strResult += '        <bridge_chip_fw>' + handleError(err) + '</bridge_chip_fw>\n'

            # Add additional code for hierarchy of bridges for Bug # 1382323                
            strResult += '      </pci_bridge_chip>\n'

            try:
                replay = nvmlDeviceGetPcieReplayCounter(handle)
                strResult += '      <replay_counter>' + str(replay) + '</replay_counter>'
            except NVMLError as err:
                strResult += '      <replay_counter>' + handleError(err) + '</replay_counter>'

            try:
                tx_bytes = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)
                strResult += '      <tx_util>' + str(tx_bytes) + ' KB/s' + '</tx_util>'
            except NVMLError as err:
                strResult += '      <tx_util>' + handleError(err) + '</tx_util>'

            try:
                rx_bytes = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)
                strResult += '      <rx_util>' + str(rx_bytes) + ' KB/s' + '</rx_util>'
            except NVMLError as err:
                strResult += '      <rx_util>' + handleError(err) + '</rx_util>'


            strResult += '    </pci>\n'

            try:
                fan = str(nvmlDeviceGetFanSpeed(handle)) + ' %'
            except NVMLError as err:
                fan = handleError(err)
            strResult += '    <fan_speed>' + fan + '</fan_speed>\n'

            try:
                perfState = nvmlDeviceGetPowerState(handle)
                perfStateStr = 'P%s' % perfState
            except NVMLError as err:
                perfStateStr = handleError(err)
            strResult += '    <performance_state>' + perfStateStr + '</performance_state>\n'

            strResult += GetClocksThrottleReasons(handle);

            try:
                memInfo = nvmlDeviceGetMemoryInfo(handle)
                mem_total = str(memInfo.total / 1024 / 1024) + ' MiB'
                mem_used = str(memInfo.used / 1024 / 1024) + ' MiB'
                mem_free = str(memInfo.total / 1024 / 1024 - memInfo.used / 1024 / 1024) + ' MiB'
            except NVMLError as err:
                error = handleError(err)
                mem_total = error
                mem_used = error
                mem_free = error

            strResult += '    <fb_memory_usage>\n'
            strResult += '      <total>' + mem_total + '</total>\n'
            strResult += '      <used>' + mem_used + '</used>\n'
            strResult += '      <free>' + mem_free + '</free>\n'
            strResult += '    </fb_memory_usage>\n'

            try:
                memInfo = nvmlDeviceGetBAR1MemoryInfo(handle)
                mem_total = str(memInfo.bar1Total / 1024 / 1024) + ' MiB'
                mem_used = str(memInfo.bar1Used / 1024 / 1024) + ' MiB'
                mem_free = str(memInfo.bar1Total / 1024 / 1024 - memInfo.bar1Used / 1024 / 1024) + ' MiB'
            except NVMLError as err:
                error = handleError(err)
                mem_total = error
                mem_used = error
                mem_free = error

            strResult += '    <bar1_memory_usage>\n'
            strResult += '      <total>' + mem_total + '</total>\n'
            strResult += '      <used>' + mem_used + '</used>\n'
            strResult += '      <free>' + mem_free + '</free>\n'
            strResult += '    </bar1_memory_usage>\n'
            
            try:
                mode = nvmlDeviceGetComputeMode(handle)
                if mode == NVML_COMPUTEMODE_DEFAULT:
                    modeStr = 'Default'
                elif mode == NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
                    modeStr = 'Exclusive Thread'
                elif mode == NVML_COMPUTEMODE_PROHIBITED:
                    modeStr = 'Prohibited'
                elif mode == NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
                    modeStr = 'Exclusive_Process'
                else:
                    modeStr = 'Unknown'
            except NVMLError as err:
                modeStr = handleError(err)

            strResult += '    <compute_mode>' + modeStr + '</compute_mode>\n'

            try:
                util = nvmlDeviceGetUtilizationRates(handle)
                gpu_util = str(util.gpu) + ' %'
                mem_util = str(util.memory) + ' %'
            except NVMLError as err:
                error = handleError(err)
                gpu_util = error
                mem_util = error
            
            strResult += '    <utilization>\n'
            strResult += '      <gpu_util>' + gpu_util + '</gpu_util>\n'
            strResult += '      <memory_util>' + mem_util + '</memory_util>\n'

            try:
                (util_int, ssize) = nvmlDeviceGetEncoderUtilization(handle)
                encoder_util = str(util_int) + ' %'
            except NVMLError as err:
                error = handleError(err)
                encoder_util = error

            strResult += '      <encoder_util>' + encoder_util + '</encoder_util>\n'

            try:
                (util_int, ssize) = nvmlDeviceGetDecoderUtilization(handle)
                decoder_util = str(util_int) + ' %'
            except NVMLError as err:
                error = handleError(err)
                decoder_util = error

            strResult += '      <decoder_util>' + decoder_util + '</decoder_util>\n'

            strResult += '    </utilization>\n'
            
            try:
                (current, pending) = nvmlDeviceGetEccMode(handle)
                curr_str = 'Enabled' if (current != 0) else 'Disabled'
                pend_str = 'Enabled' if (pending != 0) else 'Disabled'
            except NVMLError as err:
                error = handleError(err)
                curr_str = error
                pend_str = error

            strResult += '    <ecc_mode>\n'
            strResult += '      <current_ecc>' + curr_str + '</current_ecc>\n'
            strResult += '      <pending_ecc>' + pend_str + '</pending_ecc>\n'
            strResult += '    </ecc_mode>\n'

            strResult += '    <ecc_errors>\n'
            strResult += GetEccStr(handle)
            strResult += '    </ecc_errors>\n'

            strResult += '    <retired_pages>\n'
            strResult += GetRetiredPagesStr(handle)
            strResult += '    </retired_pages>\n'
            
            try:
                temp = str(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)) + ' C'
            except NVMLError as err:
                temp = handleError(err)

            strResult += '    <temperature>\n'
            strResult += '      <gpu_temp>' + temp + '</gpu_temp>\n'

            try:
                temp = str(nvmlDeviceGetTemperatureThreshold(handle, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)) + ' C'
            except NVMLError as err:
                temp = handleError(err)

            strResult += '      <gpu_temp_max_threshold>' + temp + '</gpu_temp_max_threshold>\n'

            try:
                temp = str(nvmlDeviceGetTemperatureThreshold(handle, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)) + ' C'
            except NVMLError as err:
                temp = handleError(err)

            strResult += '      <gpu_temp_slow_threshold>' + temp + '</gpu_temp_slow_threshold>\n'
            strResult += '    </temperature>\n'

            strResult += '    <power_readings>\n'
            try:
                perfState = 'P' + str(nvmlDeviceGetPowerState(handle))
            except NVMLError as err:
                perfState = handleError(err)
            strResult += '      <power_state>%s</power_state>\n' % perfState
            try:
                powMan = nvmlDeviceGetPowerManagementMode(handle)
                powManStr = 'Supported' if powMan != 0 else 'N/A'
            except NVMLError as err:
                powManStr = handleError(err)
            strResult += '      <power_management>' + powManStr + '</power_management>\n'
            try:
                powDraw = (nvmlDeviceGetPowerUsage(handle) / 1000.0)
                powDrawStr = '%.2f W' % powDraw
            except NVMLError as err:
                powDrawStr = handleError(err)
            strResult += '      <power_draw>' + powDrawStr + '</power_draw>\n'
            try:
                powLimit = (nvmlDeviceGetPowerManagementLimit(handle) / 1000.0)
                powLimitStr = '%.2f W' % powLimit
            except NVMLError as err:
                powLimitStr = handleError(err)
            strResult += '      <power_limit>' + powLimitStr + '</power_limit>\n'
            try:
                powLimit = (nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0)
                powLimitStr = '%.2f W' % powLimit
            except NVMLError as err:
                powLimitStr = handleError(err)
            strResult += '      <default_power_limit>' + powLimitStr + '</default_power_limit>\n'

            try:
                enforcedPowLimit = (nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
                enforcedPowLimitStr = '%.2f W' % enforcedPowLimit
            except NVMLError as err:
                enforcedPowLimitStr = handleError(err)

            strResult += '      <enforced_power_limit>' + enforcedPowLimitStr + '</enforced_power_limit>\n'

            try:
                powLimit = nvmlDeviceGetPowerManagementLimitConstraints(handle)
                powLimitStrMin = '%.2f W' % (powLimit[0] / 1000.0)
                powLimitStrMax = '%.2f W' % (powLimit[1] / 1000.0)
            except NVMLError as err:
                error = handleError(err)
                powLimitStrMin = error
                powLimitStrMax = error
            strResult += '      <min_power_limit>' + powLimitStrMin + '</min_power_limit>\n'
            strResult += '      <max_power_limit>' + powLimitStrMax + '</max_power_limit>\n'

            strResult += '    </power_readings>\n'

            strResult += '    <clocks>\n'
            try:
                graphics = str(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
            except NVMLError as err:
                graphics = handleError(err)
            strResult += '      <graphics_clock>' +graphics + '</graphics_clock>\n'
            try:
                sm = str(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)) + ' MHz'
            except NVMLError as err:
                sm = handleError(err)
            strResult += '      <sm_clock>' + sm + '</sm_clock>\n'
            try:
                mem = str(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)) + ' MHz'
            except NVMLError as err:
                mem = handleError(err)
            strResult += '      <mem_clock>' + mem + '</mem_clock>\n'
            strResult += '    </clocks>\n'

            strResult += '    <applications_clocks>\n'
            try:
                graphics = str(nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
            except NVMLError as err:
                graphics = handleError(err)
            strResult += '      <graphics_clock>' +graphics + '</graphics_clock>\n'
            try:
                mem = str(nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_MEM)) + ' MHz'
            except NVMLError as err:
                mem = handleError(err)
            strResult += '      <mem_clock>' + mem + '</mem_clock>\n'
            strResult += '    </applications_clocks>\n'
            
            strResult += '    <default_applications_clocks>\n'
            try:
                graphics = str(nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
            except NVMLError as err:
                graphics = handleError(err)
            strResult += '      <graphics_clock>' +graphics + '</graphics_clock>\n'
            try:
                mem = str(nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_MEM)) + ' MHz'
            except NVMLError as err:
                mem = handleError(err)
            strResult += '      <mem_clock>' + mem + '</mem_clock>\n'
            strResult += '    </default_applications_clocks>\n'

            strResult += '    <max_clocks>\n'
            try:
                graphics = str(nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
            except NVMLError as err:
                graphics = handleError(err)
            strResult += '      <graphics_clock>' + graphics + '</graphics_clock>\n'
            try:
                sm = str(nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM)) + ' MHz'
            except NVMLError as err:
                sm = handleError(err)
            strResult += '      <sm_clock>' + sm + '</sm_clock>\n'
            try:
                mem = str(nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_MEM)) + ' MHz'
            except NVMLError as err:
                mem = handleError(err)
            strResult += '      <mem_clock>' + mem + '</mem_clock>\n'
            strResult += '    </max_clocks>\n'
            
            strResult += '    <clock_policy>\n'
            try:
                boostedState, boostedDefaultState = nvmlDeviceGetAutoBoostedClocksEnabled(handle)
                if boostedState == NVML_FEATURE_DISABLED:
                    autoBoostStr = "Off"
                else:
                    autoBoostStr = "On"
                
                if boostedDefaultState == NVML_FEATURE_DISABLED:
                    autoBoostDefaultStr = "Off"
                else:
                    autoBoostDefaultStr = "On"
                
            except NVMLError_NotSupported:
                autoBoostStr = "N/A"
                autoBoostDefaultStr = "N/A"
            except NVMLError as err:
                autoBoostStr = handleError(err)
                autoBoostDefaultStr = handleError(err)
                pass
            strResult += '      <auto_boost>' + autoBoostStr + '</auto_boost>\n'
            strResult += '      <auto_boost_default>' + autoBoostDefaultStr + '</auto_boost_default>\n'
            strResult += '    </clock_policy>\n'

            try:
                memClocks = nvmlDeviceGetSupportedMemoryClocks(handle)
                strResult += '    <supported_clocks>\n'

                for m in memClocks:
                    strResult += '      <supported_mem_clock>\n'
                    strResult += '        <value>%d MHz</value>\n' % m
                    try:
                        clocks = nvmlDeviceGetSupportedGraphicsClocks(handle, m)
                        for c in clocks:
                            strResult += '        <supported_graphics_clock>%d MHz</supported_graphics_clock>\n' % c
                    except NVMLError as err:
                        strResult += '        <supported_graphics_clock>%s</supported_graphics_clock>\n' % handleError(err)
                    strResult += '      </supported_mem_clock>\n'

                strResult += '    </supported_clocks>\n'
            except NVMLError as err:
                strResult += '    <supported_clocks>' + handleError(err) + '</supported_clocks>\n'

            try:
                procs = nvmlDeviceGetComputeRunningProcesses(handle)
                strResult += '    <processes>\n'
             
                for p in procs:
                    try:
                        name = str(nvmlSystemGetProcessName(p.pid))
                    except NVMLError as err:
                        if (err.value == NVML_ERROR_NOT_FOUND):
                            # probably went away
                            continue
                        else:
                            name = handleError(err)
                    
                    strResult += '    <process_info>\n'
                    strResult += '      <pid>%d</pid>\n' % p.pid
                    strResult += '      <process_name>' + name + '</process_name>\n'

                    if (p.usedGpuMemory == None):
                        mem = 'N\A'
                    else:
                        mem = '%d MiB' % (p.usedGpuMemory / 1024 / 1024)
                    strResult += '      <used_memory>' + mem + '</used_memory>\n'
                    strResult += '    </process_info>\n'
                
                strResult += '    </processes>\n'
            except NVMLError as err:
                strResult += '    <processes>' + handleError(err) + '</processes>\n'
            

            try:
                pids = nvmlDeviceGetAccountingPids(handle)
                strResult += '    <accounted_processes>\n'
             
                for pid in pids :
                    try:
                        stats = nvmlDeviceGetAccountingStats(handle, pid) 
                        gpuUtilization = "%d %%" % stats.gpuUtilization
                        memoryUtilization = "%d %%" % stats.memoryUtilization
                        if (stats.maxMemoryUsage == None):
                            maxMemoryUsage = 'N\A'
                        else:
                            maxMemoryUsage = '%d MiB' % (stats.maxMemoryUsage / 1024 / 1024)
                        time = "%d ms" % stats.time
                        is_running = "%d" % stats.isRunning
                    except NVMLError as err:
                        if (err.value == NVML_ERROR_NOT_FOUND):
                            # probably went away
                            continue
                        err = handleError(err)
                        gpuUtilization = err
                        memoryUtilization = err
                        maxMemoryUsage = err
                        time = err
                        is_running = err
                    
                    strResult += '    <accounted_process_info>\n'
                    strResult += '      <pid>%d</pid>\n' % pid
                    strResult += '      <gpu_util>' + gpuUtilization + '</gpu_util>\n'
                    strResult += '      <memory_util>' + memoryUtilization + '</memory_util>\n'
                    strResult += '      <max_memory_usage>' + maxMemoryUsage+ '</max_memory_usage>\n'
                    strResult += '      <time>' + time + '</time>\n'
                    strResult += '      <is_running>' + is_running + '</is_running>\n'
                    strResult += '    </accounted_process_info>\n'
                
                strResult += '    </accounted_processes>\n'
            except NVMLError as err:
                strResult += '    <accounted_processes>' + handleError(err) + '</accounted_processes>\n'

            strResult += '  </gpu>\n'
            
        strResult += '</nvidia_smi_log>\n'
        
    except NVMLError as err:
        strResult += 'nvidia_smi.py: ' + err.__str__() + '\n'
    
    nvmlShutdown()
    
    return strResult

# this is not exectued when module is imported
if __name__ == "__main__":
    print(XmlDeviceQuery())
