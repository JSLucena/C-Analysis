from tree_sitter import Language, Parser
import warnings
warnings.simplefilter("ignore", FutureWarning)

Language.build_library(
    'build/language.so',
    [
        'tree-sitter-c'
    ]
)

