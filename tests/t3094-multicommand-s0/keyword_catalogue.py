"""Independent Draft 2020-12 keyword catalogue for the T-3094 subset.

The inventory is transcribed from the Core and Validation vocabularies, not
from sslm_convert_schema.py. Unknown extension keys are an explicit case.
https://json-schema.org/draft/2020-12/json-schema-core
https://json-schema.org/draft/2020-12/json-schema-validation
"""

ANNOTATIONS = frozenset({
    "title", "description", "default", "examples", "deprecated", "readOnly",
    "writeOnly", "$comment",
})
ROOT_META = frozenset({"$schema", "$id"})
STRUCTURAL = frozenset({
    "type", "enum", "const", "properties", "required", "additionalProperties",
    "items", "minItems", "prefixItems", "contains", "maxItems", "uniqueItems",
    "minContains", "maxContains", "patternProperties", "propertyNames",
    "maxProperties", "minProperties", "dependentRequired", "dependentSchemas",
    "unevaluatedProperties", "unevaluatedItems", "allOf", "anyOf", "oneOf",
    "not", "if", "then", "else", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern",
    "format", "contentEncoding", "contentMediaType", "contentSchema",
    "$ref", "$dynamicRef", "$anchor", "$dynamicAnchor", "$defs", "$vocabulary",
    # Former spellings and the two reserved 2019-09 recursion keywords must
    # also be refused, never silently treated as annotations by this subset.
    "additionalItems", "definitions", "dependencies", "$recursiveRef", "$recursiveAnchor",
})
KEYWORDS = STRUCTURAL | ANNOTATIONS | ROOT_META | {"vendorUnknownConstraint"}
ALLOWED = {
    "root": ANNOTATIONS | ROOT_META | {"type", "properties", "required", "additionalProperties"},
    "array": ANNOTATIONS | {"type", "items", "minItems"},
    "item": ANNOTATIONS | {"type", "properties", "required", "additionalProperties"},
    "enum": ANNOTATIONS | {"type", "enum"},
    "subject": ANNOTATIONS | {"type"},
    "target": ANNOTATIONS | {"type"},
    "trigger": ANNOTATIONS | {"type"},
    "boolean": ANNOTATIONS | {"type"},
}

VALUES = {
    "title": "fixture", "description": "fixture", "default": "fixture",
    "examples": ["fixture"], "deprecated": False, "readOnly": False,
    "writeOnly": False, "$comment": "fixture",
    "type": "string", "enum": ["fire"], "const": "fire",
    "properties": {}, "required": [], "additionalProperties": False,
    "items": {"type": "string"}, "minItems": 1, "prefixItems": [],
    "contains": {"type": "string"}, "maxItems": 2, "uniqueItems": True,
    "minContains": 1, "maxContains": 2, "patternProperties": {},
    "propertyNames": {}, "maxProperties": 5, "minProperties": 1,
    "dependentRequired": {}, "dependentSchemas": {},
    "unevaluatedProperties": False, "unevaluatedItems": False,
    "allOf": [], "anyOf": [], "oneOf": [], "not": {}, "if": {},
    "then": {}, "else": {}, "minimum": 1, "maximum": 5,
    "exclusiveMinimum": 1, "exclusiveMaximum": 5, "multipleOf": 2,
    "minLength": 1, "maxLength": 8, "pattern": ".+", "format": "uri",
    "contentEncoding": "base64", "contentMediaType": "text/plain",
    "contentSchema": {}, "$ref": "#/$defs/x", "$dynamicRef": "#x",
    "$anchor": "x", "$dynamicAnchor": "x", "$defs": {},
    "$vocabulary": {}, "$schema": "https://json-schema.org/draft/2020-12/schema",
    "additionalItems": False, "definitions": {}, "dependencies": {},
    "$recursiveRef": "#", "$recursiveAnchor": True,
    "$id": "https://example.invalid/t3094-command-schema",
    "vendorUnknownConstraint": 7,
}


def node(schema, location):
    if location == "root":
        return schema
    array = schema["properties"]["commands"]
    if location == "array":
        return array
    item = array["items"]
    if location == "item":
        return item
    name = "action" if location == "enum" else (
        "depends_on_previous" if location == "boolean" else location)
    return item["properties"][name]
