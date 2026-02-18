# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'PyStatistics'
copyright = '2026, SGCX'
author = 'Hai-Shuo'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Napoleon settings (support both Google and NumPy docstring styles)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    'DESIGN.md', 'ROADMAP.md', 'Forge.md',
                    'PYSTATSBIO_CONTEXT.md']

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_title = 'PyStatistics API Reference'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    'announcement': '<a href="https://sgcx.org/technology/pystatistics/">← Back to PyStatistics overview on sgcx.org</a>',
    'source_repository': 'https://github.com/sgcx-org/pystatistics',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'light_css_variables': {
        'color-brand-primary': '#27ae60',
        'color-brand-content': '#1a7a3a',
    },
    'dark_css_variables': {
        'color-brand-primary': '#2ecc71',
        'color-brand-content': '#27ae60',
    },
}

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
