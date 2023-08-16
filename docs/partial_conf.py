discussion_channel_map = {
    "ivy.data_classes.array.array": ["933380487353872454"],
    "ivy.data_classes.container.container": ["982738042886422598"],
    "ivy.functional.ivy.activations": ["1000043490329251890"],
    "ivy.functional.ivy.compilation": ["1000043526849056808"],
    "ivy.functional.ivy.constants": ["1000043627961135224"],
    "ivy.functional.ivy.creation": ["1000043690254946374"],
    "ivy.functional.ivy.data_type": ["1000043749088436315"],
    "ivy.functional.ivy.device": ["1000043775021826229"],
    "ivy.functional.ivy.elementwise": ["1000043825085026394"],
    "ivy.functional.ivy.experimental": ["1028272402624434196"],
    "ivy.functional.ivy.extensions": ["1028272402624434196"],
    "ivy.functional.ivy.general": ["1000043859973247006"],
    "ivy.functional.ivy.gradients": ["1000043921633722509"],
    "ivy.functional.ivy.layers": ["1000043967989162005"],
    "ivy.functional.ivy.linear_algebra": ["1000044022942933112"],
    "ivy.functional.ivy.losses": ["1000044049333485648"],
    "ivy.functional.ivy.manipulation": ["1000044082489466951"],
    "ivy.functional.ivy.meta": ["1000044106959044659"],
    "ivy.functional.ivy.nest": ["1000044136000393326"],
    "ivy.functional.ivy.norms": ["1000044163070447626"],
    "ivy.functional.ivy.random": ["1000044191658815569"],
    "ivy.functional.ivy.searching": ["1000044227247484980"],
    "ivy.functional.ivy.set": ["1000044247606644786"],
    "ivy.functional.ivy.sorting": ["1000044274148184084"],
    "ivy.functional.ivy.statistical": ["1000044336479731872"],
    "ivy.functional.ivy.utility": ["1000044369044312164"],
    "ivy.stateful.activations": ["1000043360297439272"],
    "ivy.stateful.converters": ["1000043009758474310"],
    "ivy.stateful.initializers": ["1000043132706115654"],
    "ivy.stateful.layers": ["1000043206840426686"],
    "ivy.stateful.module": ["1000043315267387502"],
    "ivy.stateful.norms": ["1000043235802107936"],
    "ivy.stateful.optimizers": ["1000043277870964747"],
    "ivy.stateful.sequential": ["1000043078381473792"],
}

# Only generate docs for index.rst
# That resolved a bug of autosummary generating docs for code-block examples
# of autosummary
autosummary_generate = ["index.rst"]

skippable_method_attributes = [{"__qualname__": "_wrap_function.<locals>.new_function"}]

# Retrieve html_theme_options from docs/conf.py
from docs.conf import html_theme_options

html_theme_options["switcher"]["json_url"] = "https://unify.ai/docs/versions/ivy.json"

repo_name = "ivy"
