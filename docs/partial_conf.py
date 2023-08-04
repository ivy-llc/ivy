discussion_channel_map = {
    "ivy.data_classes.array.array": ["933380487353872454", "1028296936203235359"],
    "ivy.data_classes.container.container": [
        "982738042886422598",
        "1028297229980668015",
    ],
    "ivy.functional.ivy.activations": ["1000043490329251890", "1028298682614947850"],
    "ivy.functional.ivy.compilation": ["1000043526849056808", "1028298745726648371"],
    "ivy.functional.ivy.constants": ["1000043627961135224", "1028298780715536454"],
    "ivy.functional.ivy.creation": ["1000043690254946374", "1028298816526499912"],
    "ivy.functional.ivy.data_type": ["1000043749088436315", "1028298847950225519"],
    "ivy.functional.ivy.device": ["1000043775021826229", "1028298877998211204"],
    "ivy.functional.ivy.elementwise": ["1000043825085026394", "1028298919488278589"],
    "ivy.functional.ivy.experimental": ["1028272402624434196", "1028298957870354542"],
    "ivy.functional.ivy.extensions": ["1028272402624434196", "1028298957870354542"],
    "ivy.functional.ivy.general": ["1000043859973247006", "1028298984806170634"],
    "ivy.functional.ivy.gradients": ["1000043921633722509", "1028299026501750826"],
    "ivy.functional.ivy.layers": ["1000043967989162005", "1114796929998667856"],
    "ivy.functional.ivy.linear_algebra": ["1000044022942933112", "1114797944147808297"],
    "ivy.functional.ivy.losses": ["1000044049333485648", "1114798186276597780"],
    "ivy.functional.ivy.manipulation": ["1000044082489466951", "1114798521095311490"],
    "ivy.functional.ivy.meta": ["1000044106959044659", "1114798684606046258"],
    "ivy.functional.ivy.nest": ["1000044136000393326", "1114798982691033108"],
    "ivy.functional.ivy.norms": ["1000044163070447626", "1114799262497251379"],
    "ivy.functional.ivy.random": ["1000044191658815569", "1114799504101752973"],
    "ivy.functional.ivy.searching": ["1000044227247484980", "1114799787091431434"],
    "ivy.functional.ivy.set": ["1000044247606644786", "1114799842095546399"],
    "ivy.functional.ivy.sorting": ["1000044274148184084", "1114799886563545178"],
    "ivy.functional.ivy.statistical": ["1000044336479731872", "1114799944130375720"],
    "ivy.functional.ivy.utility": ["1000044369044312164", "1114801192548175912"],
    "ivy.stateful.activations": ["1000043360297439272", "1114801905550499900"],
    "ivy.stateful.converters": ["1000043009758474310", "1114801990573228083"],
    "ivy.stateful.initializers": ["1000043132706115654", "1114802073247158332"],
    "ivy.stateful.layers": ["1000043206840426686", "1114802113554415666"],
    "ivy.stateful.module": ["1000043315267387502", "1114801941801869333"],
    "ivy.stateful.norms": ["1000043235802107936", "1114802148690120764"],
    "ivy.stateful.optimizers": ["1000043277870964747", "1114802196752646185"],
    "ivy.stateful.sequential": ["1000043078381473792", "1114802245465284629"],
}

# Only generate docs for index.rst
# That resolved a bug of autosummary generating docs for code-block examples
# of autosummary
autosummary_generate = ["index.rst"]

skippable_method_attributes = [{"__qualname__": "_wrap_function.<locals>.new_function"}]

# Retrieve html_theme_options from docs/conf.py
#from docs.conf import html_theme_options

#html_theme_options["switcher"]["json_url"] = "https://unify.ai/docs/versions/ivy.json"

repo_name = "ivy"
