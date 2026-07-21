def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "workspace_integration: requires canonical sibling workspace checkouts",
    )
