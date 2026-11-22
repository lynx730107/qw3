def parse_tags(line):
    return [tag.strip() for tag in line.split(",") if tag.strip()]
