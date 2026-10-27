"""Verify the numerical tables embedded in the DDRHF reference Markdown."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


# Make the source checkout importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "td" and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._table is not None and self._row is not None:
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _number(value: str):
    """Parse a decimal value, accepting the common Unicode minus signs."""
    value = (
        value.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "")
        .strip()
    )
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", value)
    return float(match.group()) if match else None


def _row_by_label(rows, label: str):
    for row in rows:
        if row and row[0].strip() == label:
            return row
    raise ValueError(f"could not find row {label!r}")


def _rms(rows, model_column: int, observed_column: int = 1):
    residuals = []
    for row in rows:
        observed = _number(row[observed_column])
        predicted = _number(row[model_column])
        if observed is not None and predicted is not None:
            residuals.append((predicted - observed) ** 2)
    if not residuals:
        raise ValueError("no numeric values available for RMS calculation")
    return math.sqrt(sum(residuals) / len(residuals)), len(residuals)


def _max_abs_errors(actual, published):
    errors = {}
    for index, (actual_value, value) in enumerate(zip(actual, published)):
        if value is None and actual_value is None:
            errors[str(index)] = 0.0
        elif value is None or actual_value is None:
            errors[str(index)] = float("inf")
        else:
            errors[str(index)] = abs(float(actual_value) - float(value))
    return {"max_abs": max(errors.values(), default=0.0), "per_field": errors}


def verify(path: Path):
    parser = _TableParser()
    parser.feed(path.read_text(encoding="utf-8"))
    if len(parser.tables) < 5:
        raise ValueError(f"expected at least five HTML tables, found {len(parser.tables)}")

    table_i, table_ii, table_iii, table_iv, table_v = parser.tables[:5]
    from crmf.nuclear_matter.ddrhf import PKO_PARAMETERS, nuclear_matter_properties

    # Table I has two consecutive parameter blocks with a repeated model key.
    table_i_errors = {}
    for name in ("PKO1", "PKO2", "PKO3"):
        first = _row_by_label(table_i[1:4], name)
        second = _row_by_label(table_i[5:], name)
        published = tuple(_number(value) for value in first[1:] + second[1:])
        parameters = PKO_PARAMETERS[name]
        actual = (
            parameters.sigma_mass_MeV,
            parameters.g_sigma_at_saturation,
            parameters.g_omega_at_saturation,
            parameters.g_rho_at_zero,
            parameters.f_pi_at_zero,
            parameters.a_rho,
            parameters.a_pi,
            parameters.saturation_density_fm3,
            *parameters.sigma_density,
            *parameters.omega_density,
        )
        table_i_errors[name] = _max_abs_errors(actual, published)

    model_names = ("PKO1", "PKO2", "PKO3", "PK1", "NL3", "PKDD", "DD-ME1")
    rms_iv = {
        name: _rms(table_iv[1:], column)[0]
        for name, column in zip(model_names, range(2, 9))
    }
    rms_v = {
        name: _rms(table_v[1:], column)[0]
        for name, column in zip(model_names, range(2, 9))
    }

    table_ii_expected = {}
    for row in table_ii[1:]:
        if row and row[0] in PKO_PARAMETERS:
            table_ii_expected[row[0]] = tuple(_number(value) for value in row[1:8])

    nuclear_matter = {}
    for name in ("PKO1", "PKO2", "PKO3"):
        result = nuclear_matter_properties(name)
        actual = {
            "K": result.incompressibility_MeV,
            "rho0": result.saturation_density_fm3,
            "J": result.symmetry_energy_MeV,
            "E/A": result.binding_energy_per_particle_MeV,
            "Ms/M": result.dirac_mass_ratio,
            "MNR/M": result.nonrelativistic_mass_ratio,
            "MR/M": result.relativistic_mass_ratio,
        }
        published = dict(zip(actual, table_ii_expected[name]))
        errors = {
            key: abs(actual[key] - published[key])
            for key in actual
            if published[key] is not None
        }
        nuclear_matter[name] = {
            "values": actual,
            "published_table_ii": published,
            "abs_errors_vs_table_ii": {
                "max_abs": max(errors.values(), default=0.0),
                "per_field": errors,
            },
        }

    # Table III stores each model in rows and each nucleus in three columns.
    table_iii_values = {}
    for row in table_iii[2:]:
        if row and row[0] in {"PKO1", "PKO2", "PKO3"}:
            table_iii_values[row[0]] = {
                "O16": _number(row[1]),
                "Ca40": _number(row[4]),
                "Ca48": _number(row[7]),
                "Zr90": _number(row[10]),
                "Pb208": _number(row[12]),
            }

    table_iii_iv_differences = {}
    nucleus_labels = {
        "O16": "$^{16}O$",
        "Ca40": "$^{40}Ca$",
        "Ca48": "$^{48}Ca$",
        "Zr90": "$^{90}Zr$",
        "Pb208": "$^{208}Pb$",
    }
    for model, values in table_iii_values.items():
        for nucleus, label in nucleus_labels.items():
            iv_row = _row_by_label(table_iv[1:], label)
            # Table IV column zero is the nuclide, column one is EXP.
            iv_column = 2 + model_names.index(model)
            table_iii_iv_differences[f"{model}_{nucleus}"] = (
                values[nucleus] - _number(iv_row[iv_column])
            )

    return {
        "source": str(path),
        "html_table_count": len(parser.tables),
        "table_i_parameter_errors": table_i_errors,
        "table_iv_rms": rms_iv,
        "table_v_rms": rms_v,
        "table_iii_iv_differences": table_iii_iv_differences,
        "nuclear_matter": nuclear_matter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="full.md containing the reference tables")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    result = verify(args.source)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"HTML tables: {result['html_table_count']}")
    print("Table I maximum parameter errors:")
    for name, values in result["table_i_parameter_errors"].items():
        print(f"  {name}: {values['max_abs']:.9g}")
    print("Table IV RMS:")
    for name, value in result["table_iv_rms"].items():
        print(f"  {name}: {value:.7f}")
    print("Table V RMS:")
    for name, value in result["table_v_rms"].items():
        print(f"  {name}: {value:.7f}")
    print("Table III/Table IV differences greater than 0.01 MeV:")
    for name, value in result["table_iii_iv_differences"].items():
        if abs(value) > 0.01:
            print(f"  {name}: {value:+.6f} MeV")
    print("Table II recomputation:")
    for name, values in result["nuclear_matter"].items():
        print(f"  {name}: {values['values']}")
        print(f"    max abs error: {values['abs_errors_vs_table_ii']['max_abs']:.7g}")


if __name__ == "__main__":
    main()
