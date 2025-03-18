# global
import os
from typing import List
from .exceptions.assertions import (
    assert_valid_source,
    assert_valid_target,
)

# local
from .translations.configurations.base_translator_config import (
    BaseTranslatorConfig,
)
from .translations.configurations.source_to_frontend_translator_config import (
    SourceToFrontendTranslatorConfig,
)
from .translations.configurations.frontend_to_ivy_translator_config import (
    FrontendToIvyTranslatorConfig,
)
from .translations.configurations.ivy_to_source_translator_config import (
    IvyToSourceTranslatorConfig,
)

from . import main_config


class ConfigurationsContainer:
    def __init__(self, base_output_dir: str = "ivy_transpiled_outputs") -> None:
        # Initialize configuration
        self.configuration = {
            key: value
            for key, value in main_config.__dict__.items()
            if not key.startswith("__")
        }
        self.configuration["BASE_OUTPUT_DIR"] = base_output_dir

        supported_frontend_fws = [
            f"{fw}_frontend"
            for fw in self.configuration["SUPPORTED_S2S_SOURCES"]
            if fw != "ivy"
        ]
        self.configuration["VALID_S2S_SOURCES"] = (
            self.configuration["SUPPORTED_S2S_SOURCES"] + supported_frontend_fws
        )
        self.configuration["VALID_S2S_TARGETS"] = (
            self.configuration["SUPPORTED_S2S_TARGETS"] + supported_frontend_fws
        )

        self.translator_configurations: List[BaseTranslatorConfig] = []

    def load_translator_configurations(self, source: str, target: str):
        """
        Function to load the configuration objects for all the translators.
        """
        if source == "ivy":
            # Load all configs for ivy-to-source translation
            self.translator_configurations.append(
                IvyToSourceTranslatorConfig(
                    source=source,
                    target=target,
                    base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                )
            )
        elif "frontend" in source:
            if target == "ivy":
                # Load all configs for frontend-to-ivy translation
                self.translator_configurations.append(
                    FrontendToIvyTranslatorConfig(
                        source=source,
                        target=target,
                        base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                    )
                )
            else:
                # Load all configs for frontend-to-source translation.
                # This means we need to perform the following sub-translations:
                # 1. frontend-to-ivy translation
                # 2. ivy-to-source translation
                self.translator_configurations.extend(
                    [
                        FrontendToIvyTranslatorConfig(
                            source=f"{source}",
                            target="ivy",
                            base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                        ),
                        IvyToSourceTranslatorConfig(
                            source="ivy",
                            target=target,
                            base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                        ),
                    ]
                )
        else:
            if "frontend" in target:
                # Load all configs for source-to-frontend translation
                self.translator_configurations.append(
                    SourceToFrontendTranslatorConfig(
                        source=source,
                        target=f"{source}_frontend",
                        base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                    ),
                )
            elif target == "ivy":
                # Load all configs for source-to-ivy translation.
                # This means we need to perform the following sub-translations:
                # 1. source-to-frontend translation
                # 2. frontend-to-ivy translation
                self.translator_configurations.extend(
                    [
                        SourceToFrontendTranslatorConfig(
                            source=source,
                            target=f"{source}_frontend",
                            base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                        ),
                        FrontendToIvyTranslatorConfig(
                            source=f"{source}_frontend",
                            target=target,
                            base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                        ),
                    ]
                )
            else:
                # Load all configs for source-to-source translation.
                # This means we need to perform the following sub-translations:
                # 1. source-to-frontend translation
                # 2. frontend-to-ivy-translation
                # 3. ivy-to-source translation
                self.translator_configurations.extend(
                    [
                        SourceToFrontendTranslatorConfig(
                            source=source,
                            target=f"{source}_frontend",
                            base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                        ),
                        FrontendToIvyTranslatorConfig(
                            source=f"{source}_frontend",
                            target="ivy",
                            base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                        ),
                        IvyToSourceTranslatorConfig(
                            source="ivy",
                            target=target,
                            base_output_dir=self.configuration["BASE_OUTPUT_DIR"],
                        ),
                    ]
                )

    def inject_dependent_configurations(self, source: str, target: str):
        """
        Function to inject the configuration objects for all the transformers
        into the translator configs.

        Note: Run this after running `self.load_translator_configurations.`
        """
        if not self.translator_configurations:
            self.load_translator_configurations(source=source, target=target)

        for translator_config in self.translator_configurations:
            for index, transformer_config in enumerate(
                translator_config.transformer_configs
            ):
                # Initialize the transformer config and update the list in the translator object
                translator_config.transformer_configs[index] = transformer_config()

    def load_configurations(self, source: str, target: str):
        """
        Initializes all the configuration objects for a (source, target) pair e.g.
        for (source="torch", target="ivy"), we would need to load two translator configurations
        i.e. `SourceToFrontendTranslatorConfig`and `FrontendToIvyTranslatorConfig. For each of these
        translator config objects, further a number of transformer configurations would be loaded
        as needed for the transformers to run.
        """

        assert_valid_source(source, self.configuration["VALID_S2S_SOURCES"])
        assert_valid_target(target, self.configuration["VALID_S2S_TARGETS"])

        # Translator configurations
        self.load_translator_configurations(source=source, target=target)

        # Inject transformer configurations in the corresponding translator config objects
        self.inject_dependent_configurations(source=source, target=target)
