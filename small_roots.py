"""Recover bounded roots of polynomial equations and modular polynomials.

The univariate modular path delegates to SageMath's maintained Coppersmith
implementation.  The multivariate path implements the deterministic shift
strategy of Jochemsz and May and treats the resulting relations as a heuristic:
every candidate is checked against the original polynomial before it is
returned.  Multiple exact or modular equations can also be solved as a bounded
polynomial system when product terms prevent use of a linear solver.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import combinations, product
from math import gcd, prod
from operator import index as integer_index
from typing import Any, Literal

from sage.all import GF, Matrix, PolynomialRing, QQ, RealField, ZZ, vector


LatticeReduction = Callable[[Any], Any]
SystemAlgorithm = Literal[
    "auto",
    "groebner",
    "msolve",
    "resultants",
    "jacobian",
]
SmallRootMethod = Literal["auto", "sage", "jochemsz_may"]
PolynomialSystemMethod = Literal[
    "auto",
    "bruteforce",
    "coppersmith",
    "groebner",
    "msolve",
    "resultants",
]


class NonInvertibleLeadingCoefficient(ValueError):
    """Expose a factor found while making a modular polynomial monic."""

    def __init__(self, factor: int, modulus: int) -> None:
        self.factor = factor
        self.modulus = modulus
        super().__init__(
            "the selected leading coefficient is not invertible modulo "
            f"{modulus}; gcd = {factor}"
        )


def _exact_integer(value: Any, name: str) -> int:
    """Return an exact integer without silently truncating rationals."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return integer_index(value)
    except TypeError:
        try:
            rational = QQ(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be an integer") from error
        if rational.denominator() != 1:
            raise TypeError(f"{name} must be an integer")
        return int(rational)


def _positive_integer(value: Any, name: str) -> int:
    """Validate a strictly positive integer option."""
    normalized = _exact_integer(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_integer(value: Any, name: str) -> int:
    """Validate a non-negative integer option."""
    normalized = _exact_integer(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _normalize_bounds(bounds: Iterable[int], count: int) -> tuple[int, ...]:
    """Return one strict symmetric bound for every variable."""
    try:
        raw_bounds = tuple(bounds)
    except TypeError as error:
        raise TypeError("bounds must be an iterable of positive integers") from error
    if len(raw_bounds) != count:
        raise ValueError(f"bounds must contain exactly {count} value(s)")
    return tuple(
        _positive_integer(bound, f"bounds[{index}]")
        for index, bound in enumerate(raw_bounds)
    )


def _normalize_beta(beta: Any) -> Any:
    """Return an exact beta in the interval ``(0, 1]``."""
    try:
        normalized = QQ(beta)
    except (TypeError, ValueError) as error:
        raise TypeError("beta must be rational") from error
    if normalized <= 0 or normalized > 1:
        raise ValueError("beta must satisfy 0 < beta <= 1")
    return normalized


def _normalize_effort(
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> int:
    """Validate the shared LLL/BKZ effort convention."""
    normalized = _positive_integer(effort, "effort")
    if lattice_reduction is not None and not callable(lattice_reduction):
        raise TypeError("lattice_reduction must be callable or None")
    if lattice_reduction is not None and normalized != 1:
        raise ValueError("effort must be 1 with a custom lattice reduction")
    return normalized


def _verbose_printer(enabled: bool) -> Callable[..., None]:
    """Create a small compatibility verbosity callback."""
    if not isinstance(enabled, bool):
        raise TypeError("verbose must be a bool")
    if enabled:
        return lambda *values: print("[small_roots]", *values)
    return lambda *values: None


def _evaluate(polynomial: Any, values: Sequence[Any]) -> Any:
    """Evaluate both Sage univariate and multivariate polynomials."""
    if len(values) == 1:
        return polynomial(values[0])
    return polynomial(*values)


def _integral_value(value: Any) -> int | None:
    """Return an integer only when a Sage value is exactly integral."""
    if hasattr(value, "lift"):
        try:
            value = value.lift()
        except (TypeError, ValueError):
            pass
    try:
        rational = QQ(value)
    except (TypeError, ValueError):
        return None
    if rational.denominator() != 1:
        return None
    return int(rational)


def _polynomial_generators(polynomial: Any) -> tuple[Any, ...]:
    """Return all parent generators, including absent variables."""
    try:
        generators = tuple(polynomial.parent().gens())
    except AttributeError as error:
        raise TypeError("f must be a Sage polynomial") from error
    if not generators:
        raise ValueError("f must belong to a polynomial ring")
    return generators


def _exponent_tuple(key: Any, variable_count: int) -> tuple[int, ...]:
    """Normalize dense-univariate and multivariate dictionary exponents."""
    try:
        exponents = tuple(int(exponent) for exponent in key)
    except TypeError:
        if variable_count != 1:
            raise ValueError("a multivariate exponent must be iterable") from None
        return (int(key),)
    if len(exponents) != variable_count:
        raise ValueError("a monomial exponent has the wrong dimension")
    return exponents


def _monomial_exponents(monomial: Any, generators: Sequence[Any]) -> tuple[int, ...]:
    """Return a monomial's exponent vector in parent-generator order."""
    if len(generators) == 1:
        return (int(monomial.degree()),)
    return tuple(int(monomial.degree(variable)) for variable in generators)


def _monomial_from_exponents(
    parent: Any,
    exponents: Sequence[int],
) -> Any:
    """Construct a monomial without relying on term-order internals."""
    result = parent.one()
    for variable, exponent in zip(parent.gens(), exponents, strict=True):
        result *= variable**exponent
    return result


def _monomial_coefficient(
    polynomial: Any,
    monomial: Any,
    generators: Sequence[Any],
) -> Any:
    """Read coefficients from both dense and multivariate polynomial types."""
    try:
        return polynomial.monomial_coefficient(monomial)
    except (AttributeError, TypeError, ValueError):
        if len(generators) != 1:
            raise
        return polynomial[int(monomial.degree())]


def _system_value(
    polynomial: Any,
    variables: Sequence[Any],
    values: Sequence[int],
) -> Any:
    """Evaluate a system polynomial in its parent-generator order."""
    assignment = dict(zip(variables, values, strict=True))
    try:
        return polynomial.subs(assignment)
    except AttributeError:
        return _evaluate(polynomial, values)


def _normalize_system(
    equations: Iterable[Any],
    variables: Iterable[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Coerce a polynomial system to a common rational polynomial ring."""
    try:
        raw_equations = tuple(equations)
    except TypeError as error:
        raise TypeError("H must be an iterable of polynomials") from error
    if not raw_equations:
        raise ValueError("H must contain at least one polynomial")

    parent = raw_equations[0].parent()
    if any(equation.parent() != parent for equation in raw_equations):
        raise ValueError("all equations must have the same parent")
    parent_generators = tuple(parent.gens())

    try:
        requested_variables = tuple(variables)
    except TypeError as error:
        raise TypeError("vs must be an iterable of polynomial variables") from error
    if not requested_variables:
        raise ValueError("vs must contain at least one variable")
    if len(set(map(str, requested_variables))) != len(requested_variables):
        raise ValueError("vs must not contain duplicate variables")

    indices: list[int] = []
    for requested in requested_variables:
        matches = [
            index
            for index, generator in enumerate(parent_generators)
            if generator == requested or str(generator) == str(requested)
        ]
        if len(matches) != 1:
            raise ValueError(f"{requested} is not an unambiguous parent generator")
        indices.append(matches[0])

    rational_parent = parent.change_ring(QQ)
    rational_generators = tuple(rational_parent.gens())
    rational_equations = tuple(rational_parent(equation) for equation in raw_equations)
    rational_variables = tuple(rational_generators[index] for index in indices)

    selected = set(indices)
    for equation in rational_equations:
        for index, generator in enumerate(rational_generators):
            if index not in selected and equation.degree(generator) > 0:
                raise ValueError(f"equation depends on unlisted variable {generator}")
    return rational_equations, rational_variables


def _verified_system_roots(
    equations: Sequence[Any],
    variables: Sequence[Any],
    candidates: Iterable[Sequence[Any]],
) -> list[tuple[int, ...]]:
    """Keep only distinct integral roots of every supplied equation."""
    roots: set[tuple[int, ...]] = set()
    for candidate in candidates:
        try:
            raw_values = tuple(candidate)
        except TypeError:
            continue
        if len(raw_values) != len(variables):
            continue
        values = tuple(_integral_value(value) for value in raw_values)
        if any(value is None for value in values):
            continue
        integral = tuple(int(value) for value in values)
        if all(
            _system_value(equation, variables, integral) == 0
            for equation in equations
        ):
            roots.add(integral)
    return sorted(roots)


def solve_system_with_gb(
    H: Iterable[Any],
    vs: Iterable[Any],
    msolve: bool = False,
) -> list[tuple[int, ...]]:
    """Return all rational-integral roots of a zero-dimensional system.

    ``msolve=True`` preserves the old optional backend.  Sage must be able to
    find the external ``msolve`` binary for that mode.
    """
    if not isinstance(msolve, bool):
        raise TypeError("msolve must be a bool")
    equations, variables = _normalize_system(H, vs)
    nonzero = tuple(equation for equation in equations if equation != 0)
    if not nonzero:
        return []

    ideal = nonzero[0].parent().ideal(nonzero)
    dimension = ideal.dimension()
    if dimension != 0:
        return []

    kwargs: dict[str, Any] = {"ring": QQ, "proof": False}
    if msolve:
        kwargs["algorithm"] = "msolve"
    try:
        variety = ideal.variety(**kwargs)
    except TypeError:
        kwargs.pop("proof", None)
        variety = ideal.variety(**kwargs)
    except (OSError, RuntimeError) as error:
        if msolve:
            raise RuntimeError(
                "the msolve backend is unavailable or failed"
            ) from error
        raise

    candidates = [
        tuple(point[variable] for variable in variables)
        for point in variety
        if all(variable in point for variable in variables)
    ]
    return _verified_system_roots(equations, variables, candidates)


def _as_univariate(
    polynomial: Any,
    variable: Any,
    all_variables: Sequence[Any],
) -> Any:
    """Convert a polynomial involving one selected variable to ``QQ[u]``."""
    variable_index = all_variables.index(variable)
    univariate_parent = PolynomialRing(QQ, "small_roots_resultant_variable")
    univariate_generator = univariate_parent.gen()

    if not hasattr(polynomial, "dict"):
        return univariate_parent(QQ(polynomial))

    result = univariate_parent.zero()
    for key, coefficient in polynomial.dict().items():
        exponents = _exponent_tuple(key, len(all_variables))
        if any(
            exponent != 0
            for index, exponent in enumerate(exponents)
            if index != variable_index
        ):
            raise ValueError("polynomial is not univariate after substitution")
        result += QQ(coefficient) * univariate_generator ** exponents[variable_index]
    return result


def _common_univariate_roots(
    equations: Sequence[Any],
    variable: Any,
    all_variables: Sequence[Any],
) -> list[Any]:
    """Find every common rational root, including zero."""
    common = None
    for equation in equations:
        if equation == 0:
            continue
        polynomial = _as_univariate(equation, variable, all_variables)
        if polynomial == 0:
            continue
        if polynomial.degree() == 0:
            return []
        common = polynomial.monic() if common is None else common.gcd(polynomial)
        if common.degree() == 0:
            return []
    if common is None:
        return []
    return [root for root, _multiplicity in common.roots(ring=QQ)]


def _resultant_assignments(
    equations: Sequence[Any],
    active_variables: Sequence[Any],
    all_variables: Sequence[Any],
) -> list[dict[Any, Any]]:
    """Recursively eliminate variables with pairwise resultants."""
    if len(active_variables) == 1:
        variable = active_variables[0]
        return [
            {variable: root}
            for root in _common_univariate_roots(
                equations,
                variable,
                all_variables,
            )
        ]

    eliminated = min(
        active_variables,
        key=lambda variable: sum(
            max(0, int(equation.degree(variable)))
            for equation in equations
            if hasattr(equation, "degree")
        ),
    )
    remaining = tuple(
        variable for variable in active_variables if variable != eliminated
    )
    involving = [
        equation
        for equation in equations
        if equation != 0 and equation.degree(eliminated) > 0
    ]
    elimination_equations = [
        equation
        for equation in equations
        if equation != 0 and equation.degree(eliminated) == 0
    ]
    for left, right in combinations(involving, 2):
        resultant = left.resultant(right, eliminated)
        if resultant != 0:
            elimination_equations.append(resultant)

    unique_eliminations: list[Any] = []
    seen: set[str] = set()
    for equation in elimination_equations:
        marker = repr(equation)
        if marker not in seen:
            seen.add(marker)
            unique_eliminations.append(equation)
    if not unique_eliminations:
        return []

    partial_assignments = _resultant_assignments(
        unique_eliminations,
        remaining,
        all_variables,
    )
    assignments: list[dict[Any, Any]] = []
    for partial in partial_assignments:
        substituted = [equation.subs(partial) for equation in equations]
        for root in _common_univariate_roots(
            substituted,
            eliminated,
            all_variables,
        ):
            assignment = dict(partial)
            assignment[eliminated] = root
            assignments.append(assignment)
    return assignments


def solve_system_with_resultants(
    H: Iterable[Any],
    vs: Iterable[Any],
) -> list[tuple[int, ...]]:
    """Return all integral roots found by recursive resultant elimination."""
    equations, variables = _normalize_system(H, vs)
    assignments = _resultant_assignments(equations, variables, variables)
    candidates = [
        tuple(assignment[variable] for variable in variables)
        for assignment in assignments
        if all(variable in assignment for variable in variables)
    ]
    return _verified_system_roots(equations, variables, candidates)


def _newton_starts(bounds: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Build deterministic central and signed starts for Newton iteration."""
    starts: list[tuple[int, ...]] = [tuple(0 for _ in bounds)]
    starts.append(tuple(bound // 2 for bound in bounds))
    starts.append(tuple(-(bound // 2) for bound in bounds))
    if len(bounds) <= 8:
        for signs in product((-1, 1), repeat=len(bounds)):
            starts.append(
                tuple(
                    sign * max(1, bound // 2)
                    for sign, bound in zip(signs, bounds, strict=True)
                )
            )
    return tuple(dict.fromkeys(starts))


def solve_system_with_jacobian(
    H: Iterable[Any],
    f: Any,
    bounds: Iterable[int],
    iters: int = 100,
    prec: int = 200,
) -> list[tuple[int, ...]]:
    """Find bounded integral roots using deterministic Newton iterations.

    Singular Jacobians only discard the current equation subset and starting
    point.  Rounded candidates must satisfy every equation exactly.
    """
    raw_equations = tuple(H)
    if not raw_equations:
        raise ValueError("H must contain at least one polynomial")
    equation_variables = tuple(raw_equations[0].parent().gens())
    f_variables = _polynomial_generators(f)
    if len(equation_variables) != len(f_variables):
        raise ValueError("H and f must have the same number of variables")
    equations, variables = _normalize_system(raw_equations, equation_variables)
    normalized_bounds = _normalize_bounds(bounds, len(variables))
    iteration_count = _positive_integer(iters, "iters")
    precision = _positive_integer(prec, "prec")

    nonconstant = tuple(
        equation for equation in equations if equation != 0 and equation.degree() > 0
    )
    if len(nonconstant) < len(variables):
        return []

    field = RealField(precision)
    derivatives = {
        (index, variable): equation.derivative(variable)
        for index, equation in enumerate(nonconstant)
        for variable in variables
    }
    candidates: list[tuple[int, ...]] = []
    for indices in combinations(range(len(nonconstant)), len(variables)):
        selected = tuple(nonconstant[index] for index in indices)
        for start in _newton_starts(normalized_bounds):
            point = vector(field, start)
            failed = False
            for _ in range(iteration_count):
                values = tuple(point)
                function_values = vector(
                    field,
                    [_evaluate(equation, values) for equation in selected],
                )
                jacobian_matrix = Matrix(
                    field,
                    [
                        [
                            _evaluate(derivatives[(index, variable)], values)
                            for variable in variables
                        ]
                        for index in indices
                    ],
                )
                try:
                    step = jacobian_matrix.solve_right(function_values)
                except (ArithmeticError, ValueError, ZeroDivisionError):
                    failed = True
                    break
                point -= step
                if max(abs(value) for value in step) < field(2) ** (-(precision // 3)):
                    break
            if failed:
                continue
            rounded = tuple(int(value.round()) for value in point)
            if all(
                abs(value) < bound
                for value, bound in zip(rounded, normalized_bounds, strict=True)
            ):
                candidates.append(rounded)
    return _verified_system_roots(equations, variables, candidates)


def _independent_rows(basis: Any) -> Any:
    """Discard exact row dependencies without changing column coordinates."""
    if basis.nrows() == 0 or basis.ncols() == 0:
        raise ValueError("the shift lattice is empty")
    pivot_rows = Matrix(QQ, basis).transpose().pivots()
    if not pivot_rows:
        raise ValueError("the shift lattice has rank zero")
    return Matrix(ZZ, [basis[index] for index in pivot_rows])


def _reduce_basis(
    basis: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any:
    """Reduce a row lattice and validate the custom reducer's result."""
    basis = _independent_rows(Matrix(ZZ, basis))
    if effort > basis.nrows():
        raise ValueError(
            f"effort cannot exceed the lattice row count ({basis.nrows()})"
        )
    if lattice_reduction is not None:
        reduced = lattice_reduction(basis)
    elif effort == 1:
        reduced = basis.LLL()
    else:
        reduced = basis.BKZ(block_size=effort)

    try:
        reduced = Matrix(ZZ, reduced)
    except (TypeError, ValueError) as error:
        raise TypeError("lattice reduction must return an integer matrix") from error
    if reduced.dimensions() != basis.dimensions():
        raise ValueError("lattice reduction changed the basis dimensions")
    if reduced.rank() != basis.nrows():
        raise ValueError("lattice reduction returned a rank-deficient basis")

    coordinates: list[list[int]] = []
    for row in reduced.rows():
        try:
            representation = basis.solve_left(row)
        except ValueError as error:
            raise ValueError(
                "lattice reduction returned a row outside the input lattice span"
            ) from error
        integral_row: list[int] = []
        for coefficient in representation:
            value = _integral_value(coefficient)
            if value is None:
                raise ValueError(
                    "lattice reduction returned a row outside the input lattice"
                )
            integral_row.append(value)
        coordinates.append(integral_row)
    transformation = Matrix(ZZ, coordinates)
    if abs(int(transformation.det())) != 1:
        raise ValueError("lattice reduction changed the generated lattice")
    return reduced


def _scaled_coefficient_lattice(
    shifts: Sequence[Any],
    generators: Sequence[Any],
    bounds: Sequence[int],
) -> tuple[Any, tuple[Any, ...], tuple[int, ...]]:
    """Build an integer coefficient lattice with variable-bound scaling."""
    monomial_set = {
        monomial
        for shift in shifts
        for monomial in shift.monomials()
    }
    monomials = tuple(
        sorted(
            monomial_set,
            key=lambda monomial: _monomial_exponents(monomial, generators),
        )
    )
    if not monomials:
        raise ValueError("the shifts contain no monomials")
    factors = tuple(
        prod(
            bound**exponent
            for bound, exponent in zip(
                bounds,
                _monomial_exponents(monomial, generators),
                strict=True,
            )
        )
        for monomial in monomials
    )
    basis = Matrix(
        ZZ,
        [
            [
                ZZ(_monomial_coefficient(shift, monomial, generators)) * factor
                for monomial, factor in zip(monomials, factors, strict=True)
            ]
            for shift in shifts
        ],
    )
    return basis, monomials, factors


def _relations_from_basis(
    reduced: Any,
    monomials: Sequence[Any],
    factors: Sequence[int],
) -> tuple[Any, ...]:
    """Unscale reduced rows exactly and turn them back into polynomials."""
    relations: list[Any] = []
    for row in reduced.rows():
        relation = monomials[0].parent().zero()
        for entry, monomial, factor in zip(
            row,
            monomials,
            factors,
            strict=True,
        ):
            if entry % factor != 0:
                raise ValueError(
                    "lattice reduction produced a non-divisible scaled coefficient"
                )
            relation += ZZ(entry // factor) * monomial
        if relation != 0 and relation.degree() > 0:
            relations.append(relation)
    return tuple(relations)


def _coppersmith_relations_from_basis(
    reduced: Any,
    monomials: Sequence[Any],
    factors: Sequence[int],
    modulus: int,
    modulus_power: int,
) -> tuple[tuple[Any, ...], int]:
    """Prioritize certified-short relations and remove scalar duplicates."""
    records: list[tuple[bool, int, tuple[int, ...], Any]] = []
    threshold_squared = modulus ** (2 * modulus_power)
    for row in reduced.rows():
        coefficients: list[int] = []
        relation = monomials[0].parent().zero()
        for entry, monomial, factor in zip(
            row,
            monomials,
            factors,
            strict=True,
        ):
            if entry % factor != 0:
                raise ValueError(
                    "lattice reduction produced a non-divisible scaled coefficient"
                )
            coefficient = int(entry // factor)
            coefficients.append(coefficient)
            relation += coefficient * monomial
        if relation == 0 or relation.degree() <= 0:
            continue

        content = 0
        for coefficient in coefficients:
            content = gcd(content, abs(coefficient))
        content = max(content, 1)
        normalized = tuple(coefficient // content for coefficient in coefficients)
        first_nonzero = next(
            coefficient for coefficient in normalized if coefficient != 0
        )
        if first_nonzero < 0:
            normalized = tuple(-coefficient for coefficient in normalized)

        norm_squared = sum(int(entry) ** 2 for entry in row)
        certified = (
            norm_squared * len(monomials) < threshold_squared
        )
        records.append((certified, norm_squared, normalized, relation))

    records.sort(key=lambda record: (not record[0], record[1]))
    seen: set[tuple[int, ...]] = set()
    relations: list[Any] = []
    certified_count = 0
    for certified, _norm_squared, normalized, relation in records:
        if normalized in seen:
            continue
        seen.add(normalized)
        relations.append(relation)
        if certified:
            certified_count += 1
    return tuple(relations), certified_count


def _coefficient_lift(coefficient: Any, modulus: int) -> int:
    """Return a centered integer lift of a residue-ring coefficient."""
    if hasattr(coefficient, "lift"):
        coefficient = coefficient.lift()
    value = _exact_integer(coefficient, "polynomial coefficient") % modulus
    if 2 * value > modulus:
        value -= modulus
    return value


def _centered_polynomial_lift(polynomial: Any, modulus: int) -> Any:
    """Lift every modular coefficient to its centered integer representative."""
    modular_parent = polynomial.parent()
    integer_parent = modular_parent.change_ring(ZZ)
    generators = tuple(integer_parent.gens())
    result = integer_parent.zero()
    for key, coefficient in polynomial.dict().items():
        exponents = _exponent_tuple(key, len(generators))
        result += _coefficient_lift(coefficient, modulus) * _monomial_from_exponents(
            integer_parent,
            exponents,
        )
    return result


def _modulus(polynomial: Any) -> int | None:
    """Return the finite coefficient-ring size, or ``None`` for ``ZZ``/``QQ``."""
    base_ring = polynomial.base_ring()
    if base_ring in (ZZ, QQ):
        return None
    try:
        modulus = int(base_ring.cardinality())
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "f must be over ZZ, QQ, or a finite integer residue ring"
        ) from error
    if modulus <= 1:
        raise ValueError("the coefficient ring must have cardinality greater than one")
    return modulus


def _make_monic_lift(
    polynomial: Any,
    modulus: int,
    coefficient: Any | None = None,
) -> Any:
    """Normalize a modular polynomial and expose non-trivial gcds."""
    selected = polynomial.lc() if coefficient is None else coefficient
    selected_lift = _coefficient_lift(selected, modulus) % modulus
    common_factor = gcd(selected_lift, modulus)
    if common_factor != 1:
        raise NonInvertibleLeadingCoefficient(common_factor, modulus)
    inverse = pow(selected_lift, -1, modulus)
    return _centered_polynomial_lift(polynomial * inverse, modulus)


def _leading_monomial(polynomial: Any, generators: Sequence[Any]) -> Any:
    """Return the term-order leading monomial for dense or multivariate input."""
    if hasattr(polynomial, "lm"):
        return polynomial.lm()
    return generators[0] ** polynomial.degree()


def _jochemsz_may_shifts(
    polynomial: Any,
    modulus: int,
    m: int,
    d: int,
) -> tuple[Any, ...]:
    """Construct deterministic Jochemsz--May shift polynomials."""
    integer_polynomial = _make_monic_lift(polynomial, modulus)
    generators = tuple(integer_polynomial.parent().gens())
    leading = _leading_monomial(integer_polynomial, generators)
    power_m_monomials = tuple((integer_polynomial**m).monomials())

    monomial_sets: list[set[Any]] = []
    for k in range(m + 1):
        target_monomials = set((integer_polynomial ** (m - k)).monomials())
        selected: set[Any] = set()
        leading_power = leading**k
        for monomial in power_m_monomials:
            try:
                quotient = monomial // leading_power
            except (ArithmeticError, TypeError, ValueError):
                continue
            if quotient not in target_monomials:
                continue
            for extra in product(range(d), repeat=len(generators)):
                selected.add(
                    monomial
                    * _monomial_from_exponents(integer_polynomial.parent(), extra)
                )
        monomial_sets.append(selected)
    monomial_sets.append(set())

    shifts: list[Any] = []
    for k in range(m + 1):
        leading_power = leading**k
        for monomial in sorted(
            monomial_sets[k] - monomial_sets[k + 1],
            key=lambda value: _monomial_exponents(value, generators),
        ):
            shifts.append(
                (monomial // leading_power)
                * integer_polynomial**k
                * modulus ** (m - k)
            )
    if not shifts:
        raise ValueError("the selected parameters produced no lattice shifts")
    return tuple(shifts)


def _polynomial_power_product(
    polynomials: Sequence[Any],
    powers: Sequence[int],
) -> Any:
    """Return ``product(polynomials[i] ** powers[i])`` exactly."""
    result = polynomials[0].parent().one()
    for polynomial, exponent in zip(polynomials, powers, strict=True):
        result *= polynomial**exponent
    return result


def _simultaneous_coppersmith_shifts(
    polynomials: Sequence[Any],
    modulus: int,
    m: int,
    shift_degree: int,
) -> tuple[Any, ...]:
    """Construct generalized Jochemsz--May shifts for several equations.

    A master monomial set is formed from every mixed product of total degree
    ``m``.  For each master monomial, the largest compatible mixed power of
    the supplied equations is used.  Every resulting polynomial vanishes at a
    common modular root modulo ``modulus**m``.
    """
    if not polynomials:
        raise ValueError("at least one nonconstant polynomial is required")
    parent = polynomials[0].parent()
    generators = tuple(parent.gens())
    equation_count = len(polynomials)
    power_vectors = tuple(IIter(m, equation_count))
    products_by_power = {
        powers: _polynomial_power_product(polynomials, powers)
        for powers in power_vectors
    }
    leading_monomials = tuple(
        _leading_monomial(polynomial, generators)
        for polynomial in polynomials
    )
    leading_by_power = {
        powers: _polynomial_power_product(leading_monomials, powers)
        for powers in power_vectors
    }

    support_by_total: dict[int, set[Any]] = {
        total: set() for total in range(m + 1)
    }
    for powers, polynomial_product in products_by_power.items():
        support_by_total[sum(powers)].update(polynomial_product.monomials())
    master_monomials = support_by_total[m]
    if not master_monomials:
        raise ValueError("the selected equation powers have no monomials")

    extra_monomials = tuple(
        _monomial_from_exponents(parent, exponents)
        for exponents in IIter(shift_degree, len(generators))
    )
    shifts: list[Any] = []
    seen: set[str] = set()
    for master_monomial in sorted(
        master_monomials,
        key=lambda value: _monomial_exponents(value, generators),
    ):
        compatible: list[tuple[int, ...]] = []
        for powers in power_vectors:
            leading_product = leading_by_power[powers]
            try:
                quotient = master_monomial // leading_product
            except (ArithmeticError, TypeError, ValueError):
                continue
            if quotient * leading_product != master_monomial:
                continue
            if quotient in support_by_total[m - sum(powers)]:
                compatible.append(powers)
        if not compatible:
            continue

        maximum_power = max(map(sum, compatible))
        for powers in compatible:
            if sum(powers) != maximum_power:
                continue
            leading_product = leading_by_power[powers]
            base_shift = (
                (master_monomial // leading_product)
                * products_by_power[powers]
                * modulus ** (m - maximum_power)
            )
            for extra_monomial in extra_monomials:
                shift = extra_monomial * base_shift
                marker = repr(shift)
                if marker not in seen:
                    seen.add(marker)
                    shifts.append(shift)
    if not shifts:
        raise ValueError("the selected parameters produced no lattice shifts")
    return tuple(shifts)


def _relation_subsets(
    relations: Sequence[Any],
    variable_count: int,
) -> Iterator[tuple[Any, ...]]:
    """Yield small deterministic systems before the complete relation set."""
    pool = tuple(relations[: max(8, 2 * variable_count + 2)])
    yielded: set[tuple[int, ...]] = set()
    maximum_size = min(len(pool), variable_count + 2)
    for size in range(variable_count, maximum_size + 1):
        for indices in combinations(range(len(pool)), size):
            if indices not in yielded:
                yielded.add(indices)
                yield tuple(pool[index] for index in indices)
            if len(yielded) >= 128:
                return
    all_indices = tuple(range(len(relations)))
    if all_indices and all_indices not in yielded:
        yield tuple(relations)


def _solve_relation_system(
    relations: Sequence[Any],
    original_polynomial: Any,
    bounds: Sequence[int],
    algorithm: SystemAlgorithm,
    candidate_validator: Callable[[tuple[int, ...]], bool] | None = None,
) -> list[tuple[int, ...]]:
    """Try bounded-size consistent subsets of heuristic lattice relations."""
    if not relations:
        return []
    variables = tuple(relations[0].parent().gens())
    algorithms = ("groebner", "resultants") if algorithm == "auto" else (algorithm,)
    for selected_algorithm in algorithms:
        for subset in _relation_subsets(relations, len(variables)):
            if selected_algorithm == "groebner":
                roots = solve_system_with_gb(subset, variables)
            elif selected_algorithm == "msolve":
                roots = solve_system_with_gb(subset, variables, msolve=True)
            elif selected_algorithm == "resultants":
                roots = solve_system_with_resultants(subset, variables)
            elif selected_algorithm == "jacobian":
                roots = solve_system_with_jacobian(
                    subset,
                    original_polynomial,
                    bounds,
                )
            else:
                raise ValueError(f"unknown system algorithm {selected_algorithm!r}")
            if roots:
                if candidate_validator is None:
                    return roots
                verified = [root for root in roots if candidate_validator(root)]
                if verified:
                    return verified
    return []


def _candidate_is_modular_root(
    polynomial: Any,
    candidate: Sequence[int],
    modulus: int,
    beta: Any,
) -> bool:
    """Check divisibility by the declared unknown-factor lower bound."""
    value = _evaluate(polynomial, candidate)
    if hasattr(value, "lift"):
        value = value.lift()
    integer = _integral_value(value)
    if integer is None:
        return False
    divisor = gcd(integer, modulus)
    return divisor ** int(beta.denominator()) >= modulus ** int(beta.numerator())


def _validate_candidates(
    polynomial: Any,
    candidates: Iterable[Sequence[Any]],
    bounds: Sequence[int],
    modulus: int | None,
    beta: Any,
) -> list[tuple[int, ...]]:
    """Normalize, bound-check, verify, deduplicate, and sort candidates."""
    roots: set[tuple[int, ...]] = set()
    for candidate in candidates:
        try:
            raw_values = tuple(candidate)
        except TypeError:
            continue
        if len(raw_values) != len(bounds):
            continue
        lifted = tuple(_integral_value(value) for value in raw_values)
        if any(value is None for value in lifted):
            continue
        values = tuple(int(value) for value in lifted)
        if modulus is not None:
            centered: list[int] = []
            for value in values:
                residue = value % modulus
                if 2 * residue > modulus:
                    residue -= modulus
                centered.append(residue)
            values = tuple(centered)
        if any(
            abs(value) >= bound
            for value, bound in zip(values, bounds, strict=True)
        ):
            continue
        if modulus is None:
            valid = _evaluate(polynomial, values) == 0
        else:
            valid = _candidate_is_modular_root(
                polynomial,
                values,
                modulus,
                beta,
            )
        if valid:
            roots.add(values)
    return sorted(roots)


def _brute_force_roots(
    polynomial: Any,
    bounds: Sequence[int],
    modulus: int | None,
    beta: Any,
    limit: int,
) -> list[tuple[int, ...]]:
    """Exhaust tiny boxes as a correctness-oriented fallback."""
    search_size = prod(2 * bound - 1 for bound in bounds)
    if limit == 0 or search_size > limit:
        return []
    candidates = product(*(range(-bound + 1, bound) for bound in bounds))
    return _validate_candidates(polynomial, candidates, bounds, modulus, beta)


def _univariate_polynomial(polynomial: Any) -> Any:
    """Return a Sage univariate polynomial exposing root operations."""
    if hasattr(polynomial, "roots"):
        return polynomial
    try:
        return polynomial.univariate_polynomial()
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "f could not be converted to a univariate polynomial"
        ) from error


def _normalize_polynomial_system_bounds(
    bounds: Mapping[Any, int] | Iterable[int],
    variables: Sequence[Any],
) -> tuple[int, ...]:
    """Return bounds in parent-generator order."""
    if not isinstance(bounds, Mapping):
        return _normalize_bounds(bounds, len(variables))

    entries = tuple(bounds.items())
    selected_entries: set[int] = set()
    ordered_bounds: list[int] = []
    for variable_index, variable in enumerate(variables):
        matches = [
            index
            for index, (key, _value) in enumerate(entries)
            if key == variable or str(key) == str(variable)
        ]
        if not matches:
            raise ValueError(f"bounds is missing variable {variable}")
        if len(matches) != 1:
            raise ValueError(f"bounds contains duplicate entries for {variable}")
        entry_index = matches[0]
        selected_entries.add(entry_index)
        ordered_bounds.append(
            _positive_integer(
                entries[entry_index][1],
                f"bounds[{variable_index}]",
            )
        )
    if len(selected_entries) != len(entries):
        raise ValueError("bounds contains a key that is not a parent generator")
    return tuple(ordered_bounds)


def _normalize_bounded_polynomial_system(
    equations: Iterable[Any],
    bounds: Mapping[Any, int] | Iterable[int],
    modulus: int | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[int, ...], int | None]:
    """Coerce equations to one polynomial ring and normalize their domain."""
    try:
        raw_equations = tuple(equations)
    except TypeError as error:
        raise TypeError("equations must be an iterable of polynomials") from error
    if not raw_equations:
        raise ValueError("equations must contain at least one polynomial")

    parent = None
    for equation in raw_equations:
        try:
            candidate_parent = equation.parent()
            candidate_generators = tuple(candidate_parent.gens())
        except AttributeError:
            continue
        if candidate_generators:
            parent = candidate_parent
            break
    if parent is None:
        raise TypeError("at least one equation must be a Sage polynomial")

    try:
        normalized_equations = tuple(parent(equation) for equation in raw_equations)
    except (TypeError, ValueError) as error:
        raise ValueError("all equations must belong to one polynomial ring") from error
    variables = tuple(parent.gens())
    normalized_bounds = _normalize_polynomial_system_bounds(bounds, variables)

    inferred_modulus = _modulus(normalized_equations[0])
    if inferred_modulus is not None:
        try:
            characteristic = int(parent.base_ring().characteristic())
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError(
                "the coefficient ring has no integer characteristic"
            ) from error
        if characteristic != inferred_modulus:
            raise TypeError(
                "extension-field polynomial systems are not supported; "
                "use GF(p) or Z/nZ"
            )

    if modulus is None:
        normalized_modulus = inferred_modulus
    else:
        normalized_modulus = _positive_integer(modulus, "modulus")
        if normalized_modulus <= 1:
            raise ValueError("modulus must be greater than one")
        if (
            inferred_modulus is not None
            and normalized_modulus != inferred_modulus
        ):
            raise ValueError(
                "modulus does not match the polynomial coefficient ring"
            )
    return (
        normalized_equations,
        variables,
        normalized_bounds,
        normalized_modulus,
    )


def _value_modulo(value: Any, modulus: int) -> int:
    """Map an integer or rational value into ``Z/modulus`` exactly."""
    if hasattr(value, "lift"):
        try:
            value = value.lift()
        except (TypeError, ValueError):
            pass
    try:
        rational = QQ(value)
    except (TypeError, ValueError) as error:
        raise TypeError("an equation did not evaluate to a rational value") from error
    numerator = int(rational.numerator())
    denominator = int(rational.denominator())
    if gcd(denominator, modulus) != 1:
        raise ValueError(
            "an equation denominator is not invertible modulo the modulus"
        )
    return numerator * pow(denominator, -1, modulus) % modulus


def _polynomial_system_candidate_is_valid(
    equations: Sequence[Any],
    variables: Sequence[Any],
    values: Sequence[int],
    modulus: int | None,
) -> bool:
    """Check a candidate against every original equation."""
    for equation in equations:
        value = _system_value(equation, variables, values)
        if modulus is None:
            if value != 0:
                return False
        elif _value_modulo(value, modulus) != 0:
            return False
    return True


def _bounded_residue_representatives(
    residue: int,
    bound: int,
    modulus: int,
) -> range:
    """Return all representatives of a residue with ``abs(x) < bound``."""
    residue %= modulus
    lower = -bound + 1
    upper = bound - 1
    first_multiplier = -((-(lower - residue)) // modulus)
    last_multiplier = (upper - residue) // modulus
    if first_multiplier > last_multiplier:
        return range(0)
    first_value = residue + first_multiplier * modulus
    return range(
        first_value,
        residue + (last_multiplier + 1) * modulus,
        modulus,
    )


def _bounded_polynomial_system_results(
    equations: Sequence[Any],
    variables: Sequence[Any],
    bounds: Sequence[int],
    candidates: Iterable[Sequence[Any]],
    modulus: int | None,
    *,
    candidates_are_residues: bool = False,
) -> list[dict[Any, int]]:
    """Normalize, bound, verify, and deduplicate system candidates."""
    roots: set[tuple[int, ...]] = set()
    for candidate in candidates:
        try:
            raw_values = tuple(candidate)
        except TypeError:
            continue
        if len(raw_values) != len(variables):
            continue
        lifted = tuple(_integral_value(value) for value in raw_values)
        if any(value is None for value in lifted):
            continue
        values = tuple(int(value) for value in lifted)

        if candidates_are_residues:
            if modulus is None:
                raise ValueError("residue candidates require a modulus")
            coordinate_options = tuple(
                _bounded_residue_representatives(value, bound, modulus)
                for value, bound in zip(values, bounds, strict=True)
            )
            bounded_candidates = product(*coordinate_options)
        else:
            bounded_candidates = (values,)

        for bounded_candidate in bounded_candidates:
            if any(
                abs(value) >= bound
                for value, bound in zip(
                    bounded_candidate,
                    bounds,
                    strict=True,
                )
            ):
                continue
            if _polynomial_system_candidate_is_valid(
                equations,
                variables,
                bounded_candidate,
                modulus,
            ):
                roots.add(tuple(bounded_candidate))

    return [
        dict(zip(variables, root, strict=True))
        for root in sorted(roots)
    ]


def _brute_force_polynomial_system(
    equations: Sequence[Any],
    variables: Sequence[Any],
    bounds: Sequence[int],
    modulus: int | None,
) -> list[dict[Any, int]]:
    """Exhaust a previously size-checked integer box."""
    candidates = product(*(range(-bound + 1, bound) for bound in bounds))
    return _bounded_polynomial_system_results(
        equations,
        variables,
        bounds,
        candidates,
        modulus,
    )


def _prime_field_system_candidates(
    equations: Sequence[Any],
    modulus: int,
) -> list[tuple[Any, ...]]:
    """Solve a zero-dimensional system over a prime field."""
    field = GF(modulus)
    finite_parent = equations[0].parent().change_ring(field)
    try:
        finite_equations = tuple(
            finite_parent(equation) for equation in equations
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "equation coefficients cannot be reduced modulo the prime"
        ) from error
    finite_variables = tuple(finite_parent.gens())
    nonzero = tuple(equation for equation in finite_equations if equation != 0)
    if not nonzero:
        raise ValueError("the polynomial system is positive-dimensional")

    ideal = finite_parent.ideal(nonzero)
    if ideal.dimension() > 0:
        raise ValueError(
            "algebraic solving requires a zero-dimensional polynomial system"
        )
    try:
        variety = ideal.variety(ring=field, proof=False)
    except TypeError:
        variety = ideal.variety(ring=field)
    candidates = [
        tuple(point[variable] for variable in finite_variables)
        for point in variety
        if all(variable in point for variable in finite_variables)
    ]
    return candidates


def _require_zero_dimensional_exact_system(
    equations: Sequence[Any],
    variables: Sequence[Any],
) -> None:
    """Reject exact systems whose bounded roots need exhaustive search."""
    rational_equations, _rational_variables = _normalize_system(
        equations,
        variables,
    )
    nonzero = tuple(equation for equation in rational_equations if equation != 0)
    if not nonzero or nonzero[0].parent().ideal(nonzero).dimension() > 0:
        raise ValueError(
            "algebraic solving requires a zero-dimensional polynomial system"
        )


def _simultaneous_coppersmith_candidates(
    equations: Sequence[Any],
    bounds: Sequence[int],
    modulus: int,
    m: int,
    shift_degree: int,
    algorithm: SystemAlgorithm,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> list[tuple[int, ...]]:
    """Recover candidates from a simultaneous modular Coppersmith lattice."""
    informative_equations: list[Any] = []
    for equation in equations:
        if equation == 0:
            continue
        if int(equation.degree()) <= 0:
            if _value_modulo(equation, modulus) != 0:
                return []
            continue
        informative_equations.append(equation)
    if not informative_equations:
        raise ValueError("the modular system contains no nonconstant equation")

    integer_equations = tuple(
        _make_monic_lift(equation, modulus)
        for equation in informative_equations
    )
    shifts = _simultaneous_coppersmith_shifts(
        integer_equations,
        modulus,
        m,
        shift_degree,
    )
    integer_variables = tuple(integer_equations[0].parent().gens())
    basis, monomials, factors = _scaled_coefficient_lattice(
        shifts,
        integer_variables,
        bounds,
    )
    report(
        "Simultaneous Coppersmith lattice dimensions: "
        f"{basis.dimensions()}"
    )
    reduced = _reduce_basis(
        basis,
        effort,
        lattice_reduction,
    )
    relations, certified_count = _coppersmith_relations_from_basis(
        reduced,
        monomials,
        factors,
        modulus,
        m,
    )
    report(
        f"Reduced relations: {len(relations)} "
        f"({certified_count} satisfy the Howgrave-Graham norm bound)"
    )

    def candidate_is_valid(candidate: tuple[int, ...]) -> bool:
        if len(candidate) != len(bounds):
            return False
        if any(
            abs(value) >= bound
            for value, bound in zip(candidate, bounds, strict=True)
        ):
            return False
        return _polynomial_system_candidate_is_valid(
            integer_equations,
            integer_variables,
            candidate,
            modulus,
        )

    return _solve_relation_system(
        relations,
        integer_equations[0],
        bounds,
        algorithm,
        candidate_validator=candidate_is_valid,
    )


def solve_bounded_polynomial_system(
    equations: Iterable[Any],
    bounds: Mapping[Any, int] | Iterable[int],
    modulus: int | None = None,
    *,
    method: PolynomialSystemMethod = "auto",
    m: int = 2,
    shift_degree: int = 1,
    algorithm: SystemAlgorithm = "auto",
    lattice_reduction: LatticeReduction | None = None,
    effort: int = 1,
    verbose: bool = False,
    brute_force_limit: int = 100_000,
) -> list[dict[Any, int]]:
    """Find common bounded roots of several polynomial equations.

    Every returned assignment satisfies ``abs(value) < bound`` and every
    equation, either exactly or modulo ``modulus``.  A modulus is inferred from
    polynomials over ``GF(p)`` or ``Z/nZ``; pass it explicitly for equations
    over ``ZZ`` or ``QQ``.

    Modular systems use simultaneous Coppersmith shifts by default, including
    mixed powers of equations containing product terms.  ``m`` controls the
    maximum mixed equation power, while ``shift_degree`` adds bounded-degree
    monomial shifts.  Larger ``effort`` values select BKZ instead of LLL.
    Algebraic independence of the reduced relations is heuristic, as with
    generic multivariate Coppersmith attacks.

    Exhaustive search is only a correctness-oriented fallback for boxes no
    larger than ``brute_force_limit``.  Set that limit to zero to require the
    selected non-exhaustive method to recover the root by itself.
    """
    if method not in {
        "auto",
        "bruteforce",
        "coppersmith",
        "groebner",
        "msolve",
        "resultants",
    }:
        raise ValueError(
            "method must be 'auto', 'bruteforce', 'coppersmith', "
            "'groebner', 'msolve', or 'resultants'"
        )
    valid_algorithms = {"auto", "groebner", "msolve", "resultants", "jacobian"}
    if algorithm not in valid_algorithms:
        raise ValueError(
            "algorithm must be 'auto', 'groebner', 'msolve', "
            "'resultants', or 'jacobian'"
        )
    normalized_m = _positive_integer(m, "m")
    normalized_shift_degree = _nonnegative_integer(
        shift_degree,
        "shift_degree",
    )
    normalized_effort = _normalize_effort(effort, lattice_reduction)
    search_limit = _nonnegative_integer(
        brute_force_limit,
        "brute_force_limit",
    )
    report = _verbose_printer(verbose)
    (
        normalized_equations,
        variables,
        normalized_bounds,
        normalized_modulus,
    ) = _normalize_bounded_polynomial_system(equations, bounds, modulus)
    search_size = prod(2 * bound - 1 for bound in normalized_bounds)

    if method == "bruteforce":
        if search_limit == 0 or search_size > search_limit:
            raise ValueError(
                f"bounded search has {search_size} points, exceeding "
                f"brute_force_limit={search_limit}"
            )
        return _brute_force_polynomial_system(
            normalized_equations,
            variables,
            normalized_bounds,
            normalized_modulus,
        )

    if normalized_modulus is None:
        if method == "coppersmith":
            raise ValueError(
                "method='coppersmith' requires modular polynomial equations"
            )
        if (
            method == "auto"
            and search_limit != 0
            and search_size <= search_limit
        ):
            return _brute_force_polynomial_system(
                normalized_equations,
                variables,
                normalized_bounds,
                None,
            )

        _require_zero_dimensional_exact_system(
            normalized_equations,
            variables,
        )
        if method == "resultants":
            candidates = solve_system_with_resultants(
                normalized_equations,
                variables,
            )
        else:
            candidates = solve_system_with_gb(
                normalized_equations,
                variables,
                msolve=method == "msolve",
            )
        return _bounded_polynomial_system_results(
            normalized_equations,
            variables,
            normalized_bounds,
            candidates,
            None,
        )

    if method in {"msolve", "resultants"}:
        raise ValueError(
            f"method='{method}' only supports exact systems; use it as the "
            "Coppersmith relation algorithm instead"
        )

    if method == "groebner":
        if not ZZ(normalized_modulus).is_prime():
            raise ValueError(
                "method='groebner' requires a prime modulus; use "
                "method='coppersmith' over a composite modulus"
            )
        finite_candidates = _prime_field_system_candidates(
            normalized_equations,
            normalized_modulus,
        )
        return _bounded_polynomial_system_results(
            normalized_equations,
            variables,
            normalized_bounds,
            finite_candidates,
            normalized_modulus,
            candidates_are_residues=True,
        )

    candidates = _simultaneous_coppersmith_candidates(
        normalized_equations,
        normalized_bounds,
        normalized_modulus,
        normalized_m,
        normalized_shift_degree,
        algorithm,
        normalized_effort,
        lattice_reduction,
        report,
    )
    roots = _bounded_polynomial_system_results(
        normalized_equations,
        variables,
        normalized_bounds,
        candidates,
        normalized_modulus,
    )
    if roots:
        return roots

    if method == "auto" and ZZ(normalized_modulus).is_prime():
        try:
            finite_candidates = _prime_field_system_candidates(
                normalized_equations,
                normalized_modulus,
            )
        except ValueError:
            finite_candidates = []
        roots = _bounded_polynomial_system_results(
            normalized_equations,
            variables,
            normalized_bounds,
            finite_candidates,
            normalized_modulus,
            candidates_are_residues=True,
        )
        if roots:
            return roots

    if search_limit != 0 and search_size <= search_limit:
        report(
            "Coppersmith recovery found no verified root; checking tiny "
            "search box"
        )
        return _brute_force_polynomial_system(
            normalized_equations,
            variables,
            normalized_bounds,
            normalized_modulus,
        )
    return []


def small_roots(
    f: Any,
    bounds: Iterable[int],
    m: int = 1,
    d: int | None = None,
    algorithm: SystemAlgorithm = "groebner",
    lattice_reduction: LatticeReduction | None = None,
    verbose: bool = False,
    *,
    beta: Any = 1,
    epsilon: Any | None = None,
    effort: int = 1,
    method: SmallRootMethod = "auto",
    brute_force_limit: int = 100_000,
) -> list[tuple[int, ...]]:
    """Return verified roots ``x`` satisfying ``abs(x[i]) < bounds[i]``.

    For a modular polynomial and ``beta < 1``, a candidate is accepted only if
    ``gcd(f(x), N) >= N**beta``.  ``effort=1`` uses LLL; larger values select a
    BKZ block size for the multivariate lattice.  Tiny search boxes are
    exhaustively checked after the algebraic method fails.
    """
    generators = _polynomial_generators(f)
    normalized_bounds = _normalize_bounds(bounds, len(generators))
    normalized_m = _positive_integer(m, "m")
    normalized_effort = _normalize_effort(effort, lattice_reduction)
    normalized_beta = _normalize_beta(beta)
    search_limit = _nonnegative_integer(brute_force_limit, "brute_force_limit")
    report = _verbose_printer(verbose)

    valid_algorithms = {"auto", "groebner", "msolve", "resultants", "jacobian"}
    if algorithm not in valid_algorithms:
        raise ValueError(
            "algorithm must be 'auto', 'groebner', 'msolve', "
            "'resultants', or 'jacobian'"
        )
    if method not in {"auto", "sage", "jochemsz_may"}:
        raise ValueError("method must be 'auto', 'sage', or 'jochemsz_may'")
    if epsilon is not None:
        try:
            normalized_epsilon = QQ(epsilon)
        except (TypeError, ValueError) as error:
            raise TypeError("epsilon must be rational or None") from error
        if normalized_epsilon <= 0:
            raise ValueError("epsilon must be positive")
    else:
        normalized_epsilon = None

    modulus = _modulus(f)
    if d is None:
        degree = int(f.degree()) if f != 0 else 1
        normalized_d = max(1, degree)
    else:
        normalized_d = _positive_integer(d, "d")

    if modulus is None:
        if normalized_beta != 1:
            raise ValueError("beta is only meaningful for modular polynomials")
        if len(generators) == 1 and f != 0:
            polynomial = _univariate_polynomial(f)
            candidates = [(root,) for root, _multiplicity in polynomial.roots(ring=QQ)]
            return _validate_candidates(
                f,
                candidates,
                normalized_bounds,
                None,
                normalized_beta,
            )
        return _brute_force_roots(
            f,
            normalized_bounds,
            None,
            normalized_beta,
            search_limit,
        )

    selected_method = method
    if selected_method == "auto":
        if (
            len(generators) == 1
            and lattice_reduction is None
            and normalized_effort == 1
        ):
            selected_method = "sage"
        else:
            selected_method = "jochemsz_may"

    candidates: list[tuple[int, ...]] = []
    if selected_method == "sage":
        if len(generators) != 1:
            raise ValueError("method='sage' only supports univariate polynomials")
        if lattice_reduction is not None or normalized_effort != 1:
            raise ValueError(
                "method='sage' does not use lattice_reduction or effort"
            )
        polynomial = _univariate_polynomial(f)
        kwargs: dict[str, Any] = {
            "X": normalized_bounds[0],
            "beta": float(normalized_beta),
        }
        if normalized_epsilon is not None:
            kwargs["epsilon"] = float(normalized_epsilon)
        report("Using Sage's univariate Coppersmith implementation")
        candidates = [(root,) for root in polynomial.small_roots(**kwargs)]
    else:
        shifts = _jochemsz_may_shifts(
            f,
            modulus,
            normalized_m,
            normalized_d,
        )
        basis, monomials, factors = _scaled_coefficient_lattice(
            shifts,
            tuple(shifts[0].parent().gens()),
            normalized_bounds,
        )
        report(f"Lattice dimensions: {basis.dimensions()}")
        reduced = _reduce_basis(
            basis,
            normalized_effort,
            lattice_reduction,
        )
        relations = _relations_from_basis(reduced, monomials, factors)
        report(f"Nonconstant reduced relations: {len(relations)}")
        candidates = _solve_relation_system(
            relations,
            f,
            normalized_bounds,
            algorithm,
        )

    roots = _validate_candidates(
        f,
        candidates,
        normalized_bounds,
        modulus,
        normalized_beta,
    )
    if roots:
        return roots
    report("Algebraic recovery found no verified root; checking tiny search box")
    return _brute_force_roots(
        f,
        normalized_bounds,
        modulus,
        normalized_beta,
        search_limit,
    )


class IIter(Iterator[tuple[int, ...]]):
    """Iterate lexicographically over non-negative tuples of sum at most ``m``."""

    def __init__(self, m: int, n: int) -> None:
        self.m = _nonnegative_integer(m, "m")
        self.n = _positive_integer(n, "n")
        self._iterator = (
            values
            for values in product(range(self.m + 1), repeat=self.n)
            if sum(values) <= self.m
        )

    def __iter__(self) -> IIter:
        return self

    def __next__(self) -> tuple[int, ...]:
        return next(self._iterator)


def small_roots_jacobian(
    f: Any,
    bounds: Iterable[int],
    m: int = 1,
    t: int = 1,
    lattice_reduction: LatticeReduction | None = None,
    verbose: bool = False,
    *,
    effort: int = 1,
    brute_force_limit: int = 100_000,
) -> list[tuple[int, ...]]:
    """Use the specialized affine-linear lattice and Jacobian solver.

    The old triangular construction silently lost terms for nonlinear input.
    This version rejects that unsupported setting and builds a complete
    coefficient lattice for affine-linear polynomials.
    """
    generators = _polynomial_generators(f)
    normalized_bounds = _normalize_bounds(bounds, len(generators))
    normalized_m = _positive_integer(m, "m")
    normalized_t = _positive_integer(t, "t")
    normalized_effort = _normalize_effort(effort, lattice_reduction)
    search_limit = _nonnegative_integer(brute_force_limit, "brute_force_limit")
    report = _verbose_printer(verbose)
    modulus = _modulus(f)
    if modulus is None:
        raise TypeError("small_roots_jacobian requires a modular polynomial")
    if f == 0 or int(f.degree()) != 1 or int(f.degree(generators[0])) != 1:
        raise ValueError(
            "small_roots_jacobian requires an affine-linear polynomial "
            "whose first variable has nonzero coefficient"
        )

    first_monomial = generators[0]
    coefficient = _monomial_coefficient(f, first_monomial, generators)
    integer_polynomial = _make_monic_lift(f, modulus, coefficient)
    integer_generators = tuple(integer_polynomial.parent().gens())
    shifts: list[Any] = []
    for exponents in IIter(normalized_m, len(integer_generators)):
        polynomial_power = exponents[0]
        shift = (
            integer_polynomial**polynomial_power
            * modulus ** max(normalized_t - polynomial_power, 0)
        )
        for index in range(1, len(integer_generators)):
            shift *= integer_generators[index] ** exponents[index]
        shifts.append(shift)

    basis, monomials, factors = _scaled_coefficient_lattice(
        shifts,
        integer_generators,
        normalized_bounds,
    )
    report(f"Jacobian lattice dimensions: {basis.dimensions()}")
    reduced = _reduce_basis(
        basis,
        normalized_effort,
        lattice_reduction,
    )
    relations = _relations_from_basis(reduced, monomials, factors)
    tiny_roots = _brute_force_roots(
        f,
        normalized_bounds,
        modulus,
        QQ(1),
        search_limit,
    )
    if tiny_roots:
        return tiny_roots
    candidates = _solve_relation_system(
        relations,
        f,
        normalized_bounds,
        "jacobian",
    )
    roots = _validate_candidates(
        f,
        candidates,
        normalized_bounds,
        modulus,
        QQ(1),
    )
    if roots:
        return roots
    return _brute_force_roots(
        f,
        normalized_bounds,
        modulus,
        QQ(1),
        search_limit,
    )


__all__ = [
    "IIter",
    "NonInvertibleLeadingCoefficient",
    "small_roots",
    "small_roots_jacobian",
    "solve_bounded_polynomial_system",
    "solve_system_with_gb",
    "solve_system_with_jacobian",
    "solve_system_with_resultants",
]
