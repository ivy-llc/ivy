#################################################################################
# Copyright (c) 2020, NVIDIA Corporation.  All rights reserved.                 #
#                                                                               #
# Redistribution and use in source and binary forms, with or without            #
# modification, are permitted provided that the following conditions are met:   #
#                                                                               #
#    * Redistributions of source code must retain the above copyright notice,   #
#      this list of conditions and the following disclaimer.                    #
#    * Redistributions in binary form must reproduce the above copyright        #
#      notice, this list of conditions and the following disclaimer in the      #
#      documentation and/or other materials provided with the distribution.     #
#    * Neither the name of the NVIDIA Corporation nor the names of its          #
#      contributors may be used to endorse or promote products derived from     #
#      this software without specific prior written permission.                 #
#                                                                               #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"   #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE     #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE    #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE     #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR           #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF          #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN       #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)       #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF        #
# THE POSSIBILITY OF SUCH DAMAGE.                                               #
#################################################################################

#
# Sample script to demonstrate the usage of NVML API python bindings
#

# To Run:
# $ python ./example.py

from pynvml import *

#
# Helper function
#
def StrVirt(mode):
    if mode == NVML_GPU_VIRTUALIZATION_MODE_NONE:
        return "None";
    elif mode == NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH:
        return "Pass-Through";
    elif mode == NVML_GPU_VIRTUALIZATION_MODE_VGPU:
        return "VGPU";
    elif mode == NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU:
        return "Host VGPU";
    elif mode == NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA:
        return "Host VSGA";
    else:
        return "Unknown";

#
# Converts errors into string messages
#
def handleError(err):
    if (err.value == NVML_ERROR_NOT_SUPPORTED):
        return "N/A"
    else:
        return err.__str__()

#######
def deviceQuery():

    strResult = ''
    try:
        #
        # Initialize NVML
        #
        nvmlInit()

        strResult += '  <driver_version>' + str(nvmlSystemGetDriverVersion()) + '</driver_version>\n'

        deviceCount = nvmlDeviceGetCount()
        strResult += '  <attached_gpus>' + str(deviceCount) + '</attached_gpus>\n'

        for i in range(0, deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)

            pciInfo = nvmlDeviceGetPciInfo(handle)

            strResult += '  <gpu id="%s">\n' % pciInfo.busId

            strResult += '    <product_name>' + nvmlDeviceGetName(handle) + '</product_name>\n'

            brandNames = {NVML_BRAND_UNKNOWN         :  "Unknown",
                          NVML_BRAND_QUADRO          :  "Quadro",
                          NVML_BRAND_TESLA           :  "Tesla",
                          NVML_BRAND_NVS             :  "NVS",
                          NVML_BRAND_GRID            :  "Grid",
                          NVML_BRAND_TITAN           :  "Titan",
                          NVML_BRAND_GEFORCE         :  "GeForce",
                          NVML_BRAND_NVIDIA_VAPPS    :  "NVIDIA Virtual Applications",
                          NVML_BRAND_NVIDIA_VPC      :  "NVIDIA Virtual PC",
                          NVML_BRAND_NVIDIA_VCS      :  "NVIDIA Virtual Compute Server",
                          NVML_BRAND_NVIDIA_VWS      :  "NVIDIA RTX Virtual Workstation",
                          NVML_BRAND_NVIDIA_VGAMING  :  "NVIDIA vGaming",
                          NVML_BRAND_QUADRO_RTX      :  "Quadro RTX",
                          NVML_BRAND_NVIDIA_RTX      :  "NVIDIA RTX",
                          NVML_BRAND_NVIDIA          :  "NVIDIA",
                          NVML_BRAND_GEFORCE_RTX     :  "GeForce RTX",
                          NVML_BRAND_TITAN_RTX       :  "TITAN RTX",

            }

            try:
                # If nvmlDeviceGetBrand() succeeds it is guaranteed to be in the dictionary
                brandName = brandNames[nvmlDeviceGetBrand(handle)]
            except NVMLError as err:
                brandName = handleError(err)

            strResult += '    <product_brand>' + brandName + '</product_brand>\n'

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

            strResult += '    <gpu_virtualization_mode>\n'
            try:
                mode = StrVirt(nvmlDeviceGetVirtualizationMode(handle))
            except NVMLError as err:
                mode = handleError(err)
            strResult += '      <virtualization_mode>' + mode + '</virtualization_mode>\n'
            strResult += '    </gpu_virtualization_mode>\n'

            try:
                gridLicensableFeatures = nvmlDeviceGetGridLicensableFeatures(handle)
                if gridLicensableFeatures.isGridLicenseSupported == 1:
                    strResult += '    <vgpu_software_licensed_product>\n'
                    for i in range(gridLicensableFeatures.licensableFeaturesCount):
                        if gridLicensableFeatures.gridLicensableFeatures[i].featureState == 0:
                            if nvmlDeviceGetVirtualizationMode(handle) == NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH:
                                strResult += '        <licensed_product_name>' + 'NVIDIA Virtual Applications' + '</licensed_product_name>\n'
                                strResult += '        <license_status>' + 'Licensed' + '</license_status>\n'
                            else:
                                strResult += '        <licensed_product_name>' + gridLicensableFeatures.gridLicensableFeatures[i].productName + '</licensed_product_name>\n'
                                strResult += '        <license_status>' + 'Unlicensed' + '</license_status>\n'
                        else:
                            strResult += '        <licensed_product_name>' + gridLicensableFeatures.gridLicensableFeatures[i].productName + '</licensed_product_name>\n'
                            strResult += '        <license_status>' + 'Licensed' + '</license_status>\n'
                    strResult += '    </vgpu_software_licensed_product>\n'
            except NVMLError as err:
                gridLicensableFeatures = handleError(err)

            strResult += '  </gpu>\n'

    except NVMLError as err:
        strResult += 'example.py: ' + err.__str__() + '\n'

    nvmlShutdown()

    return strResult

# If this is not exectued when module is imported
if __name__ == "__main__":
    print(deviceQuery())

