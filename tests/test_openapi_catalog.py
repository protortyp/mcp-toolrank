import unittest

from examples.openapi_catalog import catalog_from_openapi, name_for


class OpenApiCatalogTests(unittest.TestCase):
    def test_local_references_and_operation_metadata(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/exhibits/{id}": {"$ref": "#/components/pathItems/museum~1exhibit", "summary": "Path text"},
                "x-ignored": {},
            },
            "components": {"pathItems": {"museum/exhibit": {
                "get": {"summary": "Find an exhibit", "description": "Find an exhibit"},
                "head": {"operationId": "stillOpen", "summary": "Check availability"},
            }}},
        }
        exhibit, availability = catalog_from_openapi(spec, "museum")
        self.assertEqual(exhibit["name"], "museum__get_exhibits_by_id")
        self.assertEqual(exhibit["description"], "Find an exhibit\nGET /exhibits/{id}")
        self.assertEqual(exhibit["_meta"]["toolrank/openapi"], {
            "namespace": "museum", "operationId": None, "method": "GET", "path": "/exhibits/{id}",
        })
        self.assertEqual(availability["name"], "museum__stillOpen")

    def test_duplicate_names_rejected_after_normalization(self):
        spec = {"openapi": "3.0.3", "paths": {
            "/exhibits": {"get": {"operationId": "exhibits/list"}},
            "/galleries": {"get": {"operationId": "exhibits_list"}},
        }}
        with self.assertRaisesRegex(ValueError, "duplicate catalog name"):
            catalog_from_openapi(spec, "museum")

    def test_external_and_cyclic_references_rejected(self):
        for ref, message in (("./external.yaml", "local JSON Pointers"),
                             ("#/paths/~1exhibits", "cyclic path item")):
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, message):
                catalog_from_openapi(
                    {"openapi": "3.1.2", "paths": {"/exhibits": {"$ref": ref}}}, "museum"
                )

    def test_names_stay_distinct_under_length_limit(self):
        common = "list" * 100
        one = name_for("museum", common + "Paintings", "get", "/paintings")
        two = name_for("museum", common + "Sculptures", "get", "/sculptures")
        self.assertEqual(len(one), 128)
        self.assertNotEqual(one, two)
