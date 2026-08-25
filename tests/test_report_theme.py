from __future__ import annotations

import unittest

from plotly import graph_objects as go

from src.analysis.report_theme import (
    plotly_config,
    polish_plotly_figure,
    report_css,
    report_toolbar,
)


class ReportThemeTests(unittest.TestCase):
    def test_toolbar_escapes_run_label_and_exposes_print_action(self):
        toolbar = report_toolbar("run <unsafe>")
        self.assertIn("run &lt;unsafe&gt;", toolbar)
        self.assertNotIn("run <unsafe>", toolbar)
        self.assertIn("window.print()", toolbar)

    def test_stylesheet_is_responsive_and_print_friendly(self):
        css = report_css()
        self.assertIn("@media (max-width: 720px)", css)
        self.assertIn("@media print", css)
        self.assertIn(".metrics", css)
        self.assertIn("table.data", css)

    def test_plotly_polish_does_not_change_trace_values(self):
        figure = go.Figure(go.Scatter(x=[1, 2], y=[3.5, 4.5]))
        before_x = tuple(figure.data[0].x)
        before_y = tuple(figure.data[0].y)
        returned = polish_plotly_figure(figure, height=420)
        self.assertIs(returned, figure)
        self.assertEqual(tuple(figure.data[0].x), before_x)
        self.assertEqual(tuple(figure.data[0].y), before_y)
        self.assertEqual(figure.layout.height, 420)

    def test_plotly_config_is_offline_export_ready(self):
        config = plotly_config()
        self.assertFalse(config["displaylogo"])
        self.assertTrue(config["responsive"])
        self.assertEqual(config["toImageButtonOptions"]["format"], "png")


if __name__ == "__main__":
    unittest.main()
