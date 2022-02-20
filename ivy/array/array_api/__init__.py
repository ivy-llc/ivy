# local
from ivy.array.array_api.arithmetic_operators import ArrayWithArrayAPIArithmeticOperators
from ivy.array.array_api.array_operators import ArrayWithArrayAPIArrayOperators
from ivy.array.array_api.attributes import ArrayWithArrayAPIAttributes
from ivy.array.array_api.bitwise_operators import ArrayWithArrayAPIBitwiseOperators
from ivy.array.array_api.comparison_operators import ArrayWithArrayAPIComparisonOperators
from ivy.array.array_api.inplace_operators import ArrayWithArrayAPIInplaceOperators
from ivy.array.array_api.reflected_operators import ArrayWithArrayAPIReflectedOperators


class ArrayWithArrayAPI(ArrayWithArrayAPIArithmeticOperators,
                        ArrayWithArrayAPIArrayOperators,
                        ArrayWithArrayAPIAttributes,
                        ArrayWithArrayAPIBitwiseOperators,
                        ArrayWithArrayAPIComparisonOperators,
                        ArrayWithArrayAPIInplaceOperators,
                        ArrayWithArrayAPIReflectedOperators):
    pass
