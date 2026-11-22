import csv


def _clean(field):
    field = field.strip()
    if len(field) >= 2 and field[0] == '"' and field[-1] == '"':
        field = field[1:-1].replace('""', '"')
    return field.strip()


def parse_tags(line):
    return [field for field in (_clean(f) for f in next(csv.reader([line]))) if field]