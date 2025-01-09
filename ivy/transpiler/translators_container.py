# global
from types import FunctionType, MethodType
from typing import List, Optional, Union

# local
from .translations.translator import Translator
from .configurations_container import ConfigurationsContainer
from .utils.cache_utils import Cacher
from .utils.logging_utils import Logger


class TranslatorsContainer:
    def __init__(self, configurations_container: ConfigurationsContainer) -> None:
        self.configurations_container: ConfigurationsContainer = (
            configurations_container
        )
        self.translators: List[Translator] = []

    def load_translators(
        self,
        cacher: Optional[Cacher] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        """Loads all the translators needed for a given (source, target) pair e.g.
        for (souce="torch", target="ivy"), we would need to run two translators
        i.e. SourceToFrontendTranslator and FrontendToIvyTranslator."""
        for (
            translator_config
        ) in self.configurations_container.translator_configurations:
            translator_config.configuration = (
                self.configurations_container.configuration
            )
            self.translators.append(
                Translator(
                    configuration=translator_config,
                    cacher=cacher if cacher else Cacher(singleton=True),
                    logger=logger if logger else Logger(),
                )
            )

    def run_translators(
        self,
        object: Union[MethodType, FunctionType, type],
        reuse_existing: bool = True,
        base_output_dir: Optional[str] = None,
    ) -> Union[MethodType, FunctionType, type]:
        """Runs the translators on the object-like."""
        if len(self.translators) == 1:
            return self.translators[0].translate(
                object, reuse_existing=reuse_existing, base_output_dir=base_output_dir
            )

        for translator in self.translators[:-1]:
            object = translator.translate(
                object, reuse_existing=reuse_existing, base_output_dir=base_output_dir
            )

        object = self.translators[-1].translate(
            object, reuse_existing=reuse_existing, base_output_dir=base_output_dir
        )

        return object
