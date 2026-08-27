"""Small-root solvers for quadratic and bounded linear equations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import lcm
from operator import index as integer_index
from typing import Any, Literal

from sage.all import Expression, Matrix, QQ, ZZ, vector


LatticeReduction = Callable[[Any], Any]
LinearMethod = Literal["auto", "direct", "kernel"]


@dataclass(frozen=True, slots=True)
class _LinearForm:
    coefficients: tuple[int, ...]
    constant: int
    modulus: int | None


def _exact_integer(value: Any, name: str) -> int:
    """Return an exact integer while rejecting lossy coercions."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _nonnegative_rational(value: Any, name: str) -> Any:
    """Return a non-negative Sage rational."""
    try:
        rational = QQ(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be rational") from error
    if rational < 0:
        raise ValueError(f"{name} must be non-negative")
    return rational


def _integral_value(value: Any, name: str) -> int:
    """Convert an exact Sage value to an integer without truncation."""
    try:
        rational = QQ(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an integer") from error
    if rational.denominator() != 1:
        raise TypeError(f"{name} must be an integer")
    return int(rational)


def _verbose_printer(enabled: bool) -> Callable[..., None]:
    """Create the deliberately small compatibility verbosity callback."""
    if not isinstance(enabled, bool):
        raise TypeError("verbose must be a bool")
    return print if enabled else lambda *args, **kwargs: None


def _evaluate_univariate(function: Any, variable: Any | None, value: int) -> int:
    """Evaluate a Sage polynomial or symbolic expression at an integer."""
    if variable is None:
        evaluated = function
    elif isinstance(function, Expression):
        evaluated = function.subs({variable: value})
    else:
        evaluated = function(value)
    return _integral_value(evaluated, "polynomial value")


def solve_quadratic_mod_power_2(
    function: Any,
    modulus: int,
    verbose: bool = False,
) -> list[int]:
    """Return every root of a degree-at-most-two polynomial modulo ``2**n``.

    Roots are lifted one bit at a time. The work is linear in the modulus bit
    length for the usual small root sets and necessarily grows with the number
    of roots that must be returned.
    """
    report = _verbose_printer(verbose)
    normalized_modulus = _exact_integer(modulus, "modulus")
    if normalized_modulus <= 0 or normalized_modulus & (normalized_modulus - 1):
        raise ValueError("modulus must be a positive power of two")

    try:
        variables = tuple(function.variables())
    except AttributeError as error:
        raise TypeError("function must be a Sage polynomial or expression") from error
    if len(variables) > 1:
        raise ValueError("function must be univariate")

    variable = variables[0] if variables else None
    if variable is not None and function != 0 and function.degree(variable) > 2:
        raise ValueError("function must have degree at most two")

    roots = [0]
    current_modulus = 1
    while current_modulus < normalized_modulus:
        lifted_modulus = current_modulus << 1
        roots = [
            candidate
            for root in roots
            for candidate in (root, root + current_modulus)
            if _evaluate_univariate(function, variable, candidate)
            % lifted_modulus
            == 0
        ]
        report(
            f"Lifted roots modulo {lifted_modulus}: "
            f"{len(roots)} candidate(s)"
        )
        current_modulus = lifted_modulus

    return sorted(roots)


def _normalize_moduli(
    modulus: Sequence[int | None] | int | None,
    equation_count: int,
) -> tuple[int | None, ...]:
    """Expand and validate per-equation moduli."""
    if modulus is None:
        values: tuple[Any, ...] = (None,) * equation_count
    else:
        try:
            scalar = _exact_integer(modulus, "modulus")
        except TypeError:
            try:
                values = tuple(modulus)
            except TypeError as error:
                raise TypeError(
                    "modulus must be an integer, None, or a sequence"
                ) from error
            if len(values) == 0:
                values = (None,) * equation_count
            elif len(values) != equation_count:
                raise ValueError("there must be one modulus per equation")
        else:
            values = (scalar,) * equation_count

    normalized: list[int | None] = []
    for index, value in enumerate(values):
        if value is None:
            normalized.append(None)
            continue
        integer = _exact_integer(value, f"modulus[{index}]")
        if integer <= 0:
            raise ValueError(f"modulus[{index}] must be positive")
        normalized.append(integer)
    return tuple(normalized)


def _normalize_bounds(
    variable_order: Sequence[Any],
    bounds: Mapping[Any, int | None],
) -> tuple[int | None, ...]:
    """Validate symmetric absolute bounds in variable order."""
    normalized: list[int | None] = []
    for variable in variable_order:
        if variable not in bounds:
            raise ValueError(f"missing bound for variable {variable}")
        bound = bounds[variable]
        if bound is None:
            normalized.append(None)
            continue
        integer = _exact_integer(bound, f"bound for {variable}")
        if integer <= 0:
            raise ValueError(f"bound for {variable} must be positive or None")
        normalized.append(integer)
    return tuple(normalized)


def _normalize_effort(
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> int:
    """Validate the shared LLL/BKZ effort convention."""
    normalized = _exact_integer(effort, "effort")
    if normalized < 1:
        raise ValueError("effort must be at least 1")
    if lattice_reduction is not None and not callable(lattice_reduction):
        raise TypeError("lattice_reduction must be callable or None")
    if lattice_reduction is not None and normalized != 1:
        raise ValueError("effort must be 1 when lattice_reduction is supplied")
    return normalized


def _reduce_integer_basis(
    basis: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any:
    """Reduce an integer row basis using a callback, LLL, or BKZ."""
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
    return reduced


def _equation_errors(
    forms: Sequence[_LinearForm],
    values: Sequence[int],
) -> tuple[int, ...]:
    """Evaluate exact errors and centered modular residues."""
    errors: list[int] = []
    for form in forms:
        residual = form.constant + sum(
            coefficient * value
            for coefficient, value in zip(
                form.coefficients,
                values,
                strict=True,
            )
        )
        if form.modulus is not None:
            residue = residual % form.modulus
            if 2 * residue > form.modulus:
                residue -= form.modulus
            residual = residue
        errors.append(residual)
    return tuple(errors)


def _validated_solution(
    forms: Sequence[_LinearForm],
    variable_order: Sequence[Any],
    bounds: Sequence[int | None],
    values: Sequence[int],
    maximal_error: Any,
    include_homogeneous_coordinate: bool,
) -> tuple[dict[Any, int], list[int]] | None:
    """Validate bounds and equations before exposing a lattice candidate."""
    if any(
        bound is not None and abs(value) > bound
        for value, bound in zip(values, bounds, strict=True)
    ):
        return None

    errors = _equation_errors(forms, values)
    if any(abs(error) > maximal_error for error in errors):
        return None

    solution = {
        variable: int(value)
        for variable, value in zip(variable_order, values, strict=True)
    }
    if include_homogeneous_coordinate:
        solution[1] = 1
    return solution, list(errors)


def _integral_matrix(rational_basis: Any, scale: Any) -> Any:
    """Apply a compatibility scale and clear all remaining denominators."""
    scaled = rational_basis * scale
    denominator = ZZ(scaled.denominator())
    return Matrix(ZZ, scaled * denominator)


def _direct_solutions(
    forms: Sequence[_LinearForm],
    variable_order: Sequence[Any],
    bounds: Sequence[int | None],
    maximal_error: Any,
    include_homogeneous_coordinate: bool,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    scale: Any,
    report: Callable[..., None],
) -> list[tuple[dict[Any, int], list[int]]]:
    """Solve by embedding bounds and equation residuals in one lattice."""
    bounded_indices = [
        index for index, bound in enumerate(bounds) if bound is not None
    ]
    modular_indices = [
        index for index, form in enumerate(forms) if form.modulus is not None
    ]

    geometry_columns = 1 + len(bounded_indices)
    row_count = 1 + len(variable_order) + len(modular_indices)
    column_count = geometry_columns + len(forms)
    basis = Matrix(QQ, row_count, column_count)

    basis[0, 0] = 1
    for equation_index, form in enumerate(forms):
        basis[0, geometry_columns + equation_index] = form.constant

    geometry_by_variable = {
        variable_index: geometry_index + 1
        for geometry_index, variable_index in enumerate(bounded_indices)
    }
    for variable_index in range(len(variable_order)):
        row_index = variable_index + 1
        if variable_index in geometry_by_variable:
            geometry_index = geometry_by_variable[variable_index]
            basis[row_index, geometry_index] = QQ(1) / bounds[variable_index]
        for equation_index, form in enumerate(forms):
            basis[row_index, geometry_columns + equation_index] = (
                form.coefficients[variable_index]
            )

    for modulus_row, equation_index in enumerate(
        modular_indices,
        start=1 + len(variable_order),
    ):
        basis[modulus_row, geometry_columns + equation_index] = forms[
            equation_index
        ].modulus

    integer_basis = _integral_matrix(basis, scale)
    report(f"Direct lattice dimensions: {integer_basis.dimensions()}")
    reduced = _reduce_integer_basis(integer_basis, effort, lattice_reduction)

    results: list[tuple[dict[Any, int], list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    variable_count = len(variable_order)
    for row in reduced.rows():
        try:
            combination = integer_basis.solve_left(row)
        except ValueError:
            continue
        if any(coefficient not in ZZ for coefficient in combination):
            continue

        homogeneous_value = int(combination[0])
        if homogeneous_value not in (-1, 1):
            continue
        sign = homogeneous_value
        values = tuple(
            sign * int(combination[index + 1])
            for index in range(variable_count)
        )
        if values in seen:
            continue
        seen.add(values)

        result = _validated_solution(
            forms,
            variable_order,
            bounds,
            values,
            maximal_error,
            include_homogeneous_coordinate,
        )
        if result is not None:
            results.append(result)
    return results


def _integer_solution_coset(
    forms: Sequence[_LinearForm],
    variable_count: int,
) -> tuple[Any, Any] | None:
    """Compute a particular solution and the projected homogeneous kernel."""
    modular_count = sum(form.modulus is not None for form in forms)
    unknown_count = variable_count + modular_count
    equations = Matrix(ZZ, len(forms), unknown_count)
    right_hand_side = vector(ZZ, [-form.constant for form in forms])

    modulus_column = variable_count
    for row_index, form in enumerate(forms):
        for column_index, coefficient in enumerate(form.coefficients):
            equations[row_index, column_index] = coefficient
        if form.modulus is not None:
            equations[row_index, modulus_column] = form.modulus
            modulus_column += 1

    if len(forms) == 0:
        return vector(ZZ, variable_count), Matrix.identity(ZZ, variable_count)

    diagonal, left_transform, right_transform = equations.smith_form()
    transformed_rhs = left_transform * right_hand_side
    rank = equations.rank()

    transformed_solution = vector(ZZ, unknown_count)
    for index in range(rank):
        diagonal_value = diagonal[index, index]
        if transformed_rhs[index] % diagonal_value:
            return None
        transformed_solution[index] = transformed_rhs[index] // diagonal_value
    if any(transformed_rhs[index] for index in range(rank, len(forms))):
        return None

    full_particular = right_transform * transformed_solution
    particular = vector(ZZ, full_particular[:variable_count])

    projected_generators = [
        list(right_transform.column(index)[:variable_count])
        for index in range(rank, unknown_count)
    ]
    projected_generators = [
        generator for generator in projected_generators if any(generator)
    ]
    if not projected_generators:
        kernel_basis = Matrix(ZZ, 0, variable_count)
    else:
        kernel_basis = Matrix(ZZ, projected_generators).row_module().basis_matrix()
    return particular, kernel_basis


def _kernel_solutions(
    forms: Sequence[_LinearForm],
    variable_order: Sequence[Any],
    bounds: Sequence[int | None],
    maximal_error: Any,
    include_homogeneous_coordinate: bool,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> list[tuple[dict[Any, int], list[int]]]:
    """Solve an exact affine system through its kernel and Kannan embedding."""
    if maximal_error != 0:
        raise ValueError("kernel method requires maximal_error=0")
    if any(bound is None for bound in bounds):
        raise ValueError("kernel method requires finite bounds for every variable")

    coset = _integer_solution_coset(forms, len(variable_order))
    if coset is None:
        return []
    particular, kernel_basis = coset

    bound_scale = lcm(*(int(bound) for bound in bounds)) if bounds else 1
    weights = [bound_scale // int(bound) for bound in bounds]

    candidate_values: list[tuple[int, ...]] = [tuple(map(int, particular))]
    if kernel_basis.nrows() > 0:
        weighted_kernel = Matrix(
            ZZ,
            [
                [row[index] * weights[index] for index in range(len(weights))]
                for row in kernel_basis.rows()
            ],
        )
        weighted_particular = vector(
            ZZ,
            [particular[index] * weights[index] for index in range(len(weights))],
        )
        target = -weighted_particular
        embedding_factor = max(1, bound_scale)

        embedded_rows = [list(row) + [0] for row in weighted_kernel.rows()]
        embedded_rows.append(list(target) + [embedding_factor])
        embedded_basis = Matrix(ZZ, embedded_rows)
        report(f"Kernel/Kannan lattice dimensions: {embedded_basis.dimensions()}")
        reduced = _reduce_integer_basis(
            embedded_basis,
            effort,
            lattice_reduction,
        )

        for row in reduced.rows():
            last_coordinate = int(row[-1])
            if abs(last_coordinate) != embedding_factor:
                continue
            sign = 1 if last_coordinate > 0 else -1
            weighted_solution = [-sign * int(entry) for entry in row[:-1]]
            values: list[int] = []
            for coordinate, weight in zip(
                weighted_solution,
                weights,
                strict=True,
            ):
                if coordinate % weight:
                    break
                values.append(coordinate // weight)
            else:
                candidate_values.append(tuple(values))

    results: list[tuple[dict[Any, int], list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for values in candidate_values:
        if values in seen:
            continue
        seen.add(values)
        result = _validated_solution(
            forms,
            variable_order,
            bounds,
            values,
            maximal_error,
            include_homogeneous_coordinate,
        )
        if result is not None:
            results.append(result)
    return results


def _solve_forms(
    forms: Sequence[_LinearForm],
    variable_order: Sequence[Any],
    bounds: Sequence[int | None],
    *,
    maximal_error: Any,
    verbose: bool,
    lattice_reduction: LatticeReduction | None,
    scale: Any,
    method: LinearMethod,
    effort: int,
    include_homogeneous_coordinate: bool,
) -> list[tuple[dict[Any, int], list[int]]]:
    """Dispatch a normalized linear problem to its lattice strategy."""
    report = _verbose_printer(verbose)
    normalized_error = _nonnegative_rational(maximal_error, "maximal_error")
    normalized_scale = _nonnegative_rational(scale, "scale")
    if normalized_scale == 0:
        raise ValueError("scale must be positive")
    normalized_effort = _normalize_effort(effort, lattice_reduction)
    if method not in ("auto", "direct", "kernel"):
        raise ValueError("method must be 'auto', 'direct', or 'kernel'")

    selected_method = method
    if selected_method == "auto":
        selected_method = (
            "kernel"
            if normalized_error == 0 and all(bound is not None for bound in bounds)
            else "direct"
        )
    report(f"Linear solver method: {selected_method}")

    if selected_method == "kernel":
        return _kernel_solutions(
            forms,
            variable_order,
            bounds,
            normalized_error,
            include_homogeneous_coordinate,
            normalized_effort,
            lattice_reduction,
            report,
        )
    return _direct_solutions(
        forms,
        variable_order,
        bounds,
        normalized_error,
        include_homogeneous_coordinate,
        normalized_effort,
        lattice_reduction,
        normalized_scale,
        report,
    )


def _ordered_used_variables(
    used_variables: set[Any],
    bounds: Mapping[Any, Any],
) -> tuple[Any, ...]:
    """Order variables deterministically using the caller's mapping order."""
    missing = used_variables - set(bounds)
    if missing:
        raise ValueError(f"missing bounds for variables: {missing}")
    return tuple(
        variable
        for variable in bounds
        if variable != 1 and variable in used_variables
    )


def solve_linear(
    fs: Sequence[Sequence[tuple[int, Any]]],
    variables: Mapping[Any, int | None],
    modulus: Sequence[int | None] | int | None,
    maximal_error: Any = 0,
    verbose: bool = False,
    lattice_reduction: LatticeReduction | None = None,
    *,
    method: LinearMethod = "auto",
    effort: int = 1,
) -> list[dict[Any, int]]:
    """Solve bounded linear forms expressed as ``(coefficient, variable)`` pairs."""
    moduli = _normalize_moduli(modulus, len(fs))
    used_variables = {
        variable
        for form in fs
        for _, variable in form
        if variable != 1
    }
    variable_order = _ordered_used_variables(used_variables, variables)
    variable_index = {
        variable: index for index, variable in enumerate(variable_order)
    }
    normalized_bounds = _normalize_bounds(variable_order, variables)

    forms: list[_LinearForm] = []
    for form_index, (form, form_modulus) in enumerate(zip(fs, moduli, strict=True)):
        coefficients = [0] * len(variable_order)
        constant = 0
        for coefficient, variable in form:
            integer_coefficient = _integral_value(
                coefficient,
                f"coefficient in equation {form_index}",
            )
            if variable == 1:
                constant += integer_coefficient
            else:
                coefficients[variable_index[variable]] += integer_coefficient
        forms.append(_LinearForm(tuple(coefficients), constant, form_modulus))

    results = _solve_forms(
        forms,
        variable_order,
        normalized_bounds,
        maximal_error=maximal_error,
        verbose=verbose,
        lattice_reduction=lattice_reduction,
        scale=QQ(1),
        method=method,
        effort=effort,
        include_homogeneous_coordinate=1 in variables,
    )
    return [solution for solution, _ in results]


def _expression_forms(
    fs: Sequence[Expression],
    variables: Mapping[Any, int | None],
    moduli: Sequence[int | None],
) -> tuple[tuple[_LinearForm, ...], tuple[Any, ...], tuple[int | None, ...]]:
    """Parse symbolic affine expressions into exact integer coefficient rows."""
    expressions: list[Expression] = []
    used_variables: set[Any] = set()
    for index, function in enumerate(fs):
        if not isinstance(function, Expression):
            raise TypeError(f"fs[{index}] must be a Sage symbolic expression")
        expression = (
            function.lhs() - function.rhs()
            if function.is_relational()
            else function
        )
        expressions.append(expression)
        used_variables.update(expression.variables())

    variable_order = _ordered_used_variables(used_variables, variables)
    normalized_bounds = _normalize_bounds(variable_order, variables)
    zero_substitution = {variable: 0 for variable in variable_order}

    forms: list[_LinearForm] = []
    for expression_index, (expression, expression_modulus) in enumerate(
        zip(expressions, moduli, strict=True)
    ):
        constant_expression = expression.subs(zero_substitution)
        constant = _integral_value(
            constant_expression,
            f"constant in equation {expression_index}",
        )
        coefficients: list[int] = []
        reconstructed: Any = constant
        for variable in variable_order:
            coefficient_expression = expression.coefficient(variable)
            if coefficient_expression.variables():
                raise ValueError(f"fs[{expression_index}] is not affine-linear")
            coefficient = _integral_value(
                coefficient_expression,
                f"coefficient of {variable} in equation {expression_index}",
            )
            coefficients.append(coefficient)
            reconstructed += coefficient * variable

        if (expression - reconstructed).expand() != 0:
            raise ValueError(f"fs[{expression_index}] is not affine-linear")
        forms.append(
            _LinearForm(tuple(coefficients), constant, expression_modulus)
        )
    return tuple(forms), variable_order, normalized_bounds


def solve_linear_expressions(
    fs: Sequence[Expression],
    variables: Mapping[Expression | int, int | None],
    modulus: Sequence[int | None] | int | None,
    maximal_error: Any = 0,
    verbose: bool = False,
    lattice_reduction: LatticeReduction | None = None,
    scale: Any = QQ(1),
    *,
    method: LinearMethod = "auto",
    effort: int = 1,
) -> list[tuple[dict[Any, int], list[int]]]:
    """Solve bounded affine Sage expressions over integers and congruences."""
    moduli = _normalize_moduli(modulus, len(fs))
    forms, variable_order, normalized_bounds = _expression_forms(
        fs,
        variables,
        moduli,
    )
    return _solve_forms(
        forms,
        variable_order,
        normalized_bounds,
        maximal_error=maximal_error,
        verbose=verbose,
        lattice_reduction=lattice_reduction,
        scale=scale,
        method=method,
        effort=effort,
        include_homogeneous_coordinate=1 in variables,
    )


def _normalize_interval(
    value: tuple[int | None, int | None] | int | None,
    variable: Any,
) -> tuple[int | None, int | None]:
    """Normalize an inclusive interval specification."""
    if value is None:
        return None, None
    try:
        endpoint = _exact_integer(value, f"bound for {variable}")
    except TypeError:
        try:
            lower, upper = value
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"bound for {variable} must be an integer, pair, or None"
            ) from error
        lower = (
            None
            if lower is None
            else _exact_integer(lower, f"lower bound for {variable}")
        )
        upper = (
            None
            if upper is None
            else _exact_integer(upper, f"upper bound for {variable}")
        )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"lower bound exceeds upper bound for {variable}")
        return lower, upper
    return min(0, endpoint), max(0, endpoint)


def solve_linear_expressions_bounds(
    fs: Sequence[Expression],
    bounds: Mapping[
        Expression | int,
        tuple[int | None, int | None] | int | None,
    ],
    modulus: Sequence[int | None] | int | None,
    maximal_error: Any = 0,
    verbose: bool = False,
    lattice_reduction: LatticeReduction | None = None,
    scale: Any = QQ(1),
    *,
    method: LinearMethod = "auto",
    effort: int = 1,
) -> list[tuple[dict[Any, int], list[int]]]:
    """Solve affine expressions whose variables have inclusive intervals."""
    intervals = {
        variable: _normalize_interval(value, variable)
        for variable, value in bounds.items()
        if variable != 1
    }

    offsets: dict[Any, int] = {}
    fixed_values: dict[Any, int] = {}
    centered_bounds: dict[Any, int | None] = {}
    substitutions: dict[Any, Any] = {}
    for variable, (lower, upper) in intervals.items():
        if lower is not None and upper is not None:
            if lower == upper:
                fixed_values[variable] = lower
                substitutions[variable] = lower
                continue
            offset = (lower + upper) // 2
            radius = max(offset - lower, upper - offset)
            offsets[variable] = offset
            centered_bounds[variable] = radius
            substitutions[variable] = variable + offset
        elif lower is not None:
            offsets[variable] = lower
            centered_bounds[variable] = None
            substitutions[variable] = variable + lower
        elif upper is not None:
            offsets[variable] = upper
            centered_bounds[variable] = None
            substitutions[variable] = variable + upper
        else:
            offsets[variable] = 0
            centered_bounds[variable] = None
            substitutions[variable] = variable

    if 1 in bounds:
        centered_bounds[1] = 1
    centered_expressions = [function.subs(substitutions) for function in fs]
    centered_results = solve_linear_expressions(
        centered_expressions,
        centered_bounds,
        modulus,
        maximal_error=maximal_error,
        verbose=verbose,
        lattice_reduction=lattice_reduction,
        scale=scale,
        method=method,
        effort=effort,
    )

    results: list[tuple[dict[Any, int], list[int]]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for centered_solution, errors in centered_results:
        solution: dict[Any, int] = {}
        if 1 in centered_solution:
            solution[1] = 1
        for variable in intervals:
            if variable in fixed_values:
                value = fixed_values[variable]
            elif variable in centered_solution:
                value = centered_solution[variable] + offsets[variable]
            else:
                continue
            lower, upper = intervals[variable]
            if lower is not None and value < lower:
                break
            if upper is not None and value > upper:
                break
            solution[variable] = value
        else:
            key = tuple(
                sorted((str(variable), value) for variable, value in solution.items())
            )
            if key not in seen:
                seen.add(key)
                results.append((solution, errors))
    return results
