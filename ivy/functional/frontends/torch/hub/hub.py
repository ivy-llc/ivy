import ivy


def load_state_dict_from_url(
    url: str,
    model_dir= None,
    map_location = None,
    progress = True,
    check_hash = False,
    file_name = None,
    weights_only= False,
):
    import torch 
    state_dict = torch.hub.load_state_dict_from_url(
        url,
        model_dir=model_dir,
        map_location=map_location,
        progress=progress,
        check_hash=check_hash,
        file_name=file_name,
        weights_only=weights_only,
    )
    
    return state_dict
    
        
