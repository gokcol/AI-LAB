"""Generate public crawler files from the app's declared page routes."""

import argparse
import ast
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "https://ai-lab.gokcol.online"
NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
# Development-only pages and the browser-specific personal notebook are not search results.
EXCLUDED = {"sandbox", "tests", "my_learning"}


def public_paths():
    paths = set()
    for node in ast.walk(ast.parse((ROOT / "gui/app.py").read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "page" and node.args):
            continue
        source = ast.literal_eval(node.args[0])
        if not (ROOT / "gui" / source).is_file():
            raise ValueError(f"Missing page: {source}")
        slug = Path(source).stem
        if slug not in EXCLUDED:
            paths.add("/" if slug == "home" else f"/{slug}")
    if "/" not in paths:
        raise ValueError("The public homepage must be included")
    return sorted(paths)


def discovery_files():
    ET.register_namespace("", NAMESPACE)
    sitemap = ET.Element(f"{{{NAMESPACE}}}urlset")
    for path in public_paths():
        url = ET.SubElement(sitemap, f"{{{NAMESPACE}}}url")
        ET.SubElement(url, f"{{{NAMESPACE}}}loc").text = ORIGIN + path
    ET.indent(sitemap, space="  ")
    return {
        "robots.txt": f"User-agent: *\nAllow: /\n\nSitemap: {ORIGIN}/sitemap.xml\n",
        "sitemap.xml": ET.tostring(sitemap, encoding="utf-8", xml_declaration=True).decode() + "\n",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed files need regeneration")
    args = parser.parse_args()
    destination = ROOT / "public"
    for name, content in discovery_files().items():
        path = destination / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                parser.exit(1, f"Run python tools/build_discovery.py to update {path}\n")
        else:
            destination.mkdir(exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(f"Crawler files ready: {len(public_paths())} public URLs")
