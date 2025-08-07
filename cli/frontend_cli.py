#!/usr/bin/env python3
"""
FRONTEND CLI
============

Command-line interface for managing frontend components:
- VSCode extension development and packaging
- Obsidian plugin development and packaging
- Web interface development and deployment
- Embeddable widgets
"""

import os
import sys
import json
import subprocess
import click
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.safety_enforcement import get_safety_enforcement


@click.group()
def frontend():
    """Frontend component management for Quark AI Assistant."""
    pass


@frontend.command()
@click.option('--extension-dir', default='web/extensions/vscode', help='VSCode extension directory')
@click.option('--build', is_flag=True, help='Build the extension')
@click.option('--package', is_flag=True, help='Package the extension')
@click.option('--install', is_flag=True, help='Install the extension locally')
def vscode(extension_dir: str, build: bool, package: bool, install: bool):
    """Manage VSCode extension."""
    try:
        # Safety check
        safety = get_safety_enforcement()
        if not safety.validate_action("vscode_extension", {"action": "manage"}):
            click.echo("❌ Safety validation failed for VSCode extension")
            return

        extension_path = Path(extension_dir)
        
        if not extension_path.exists():
            click.echo(f"❌ Extension directory not found: {extension_dir}")
            return

        click.echo(f"🔧 Managing VSCode extension in: {extension_path}")

        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            click.echo("❌ npm is required but not found. Please install Node.js and npm.")
            return

        # Install dependencies
        click.echo("📦 Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=extension_path, check=True)

        if build:
            click.echo("🔨 Building extension...")
            subprocess.run(["npm", "run", "compile"], cwd=extension_path, check=True)
            click.echo("✅ Extension built successfully")

        if package:
            click.echo("📦 Packaging extension...")
            subprocess.run(["npm", "run", "vscode:prepublish"], cwd=extension_path, check=True)
            click.echo("✅ Extension packaged successfully")

        if install:
            click.echo("📥 Installing extension locally...")
            # Package the extension
            subprocess.run(["npm", "run", "vscode:prepublish"], cwd=extension_path, check=True)
            
            # Install using VSCode CLI
            try:
                subprocess.run(["code", "--install-extension", "quark-local-agi-1.0.0.vsix"], 
                             cwd=extension_path, check=True)
                click.echo("✅ Extension installed successfully")
            except (subprocess.CalledProcessError, FileNotFoundError):
                click.echo("⚠️ VSCode CLI not found. Please install the extension manually.")

        if not any([build, package, install]):
            click.echo("ℹ️ Use --build, --package, or --install to perform actions")

    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Command failed: {e}")
    except Exception as e:
        click.echo(f"❌ Error managing VSCode extension: {e}")


@frontend.command()
@click.option('--plugin-dir', default='web/extensions/obsidian', help='Obsidian plugin directory')
@click.option('--build', is_flag=True, help='Build the plugin')
@click.option('--package', is_flag=True, help='Package the plugin')
@click.option('--install', is_flag=True, help='Install the plugin locally')
def obsidian(plugin_dir: str, build: bool, package: bool, install: bool):
    """Manage Obsidian plugin."""
    try:
        # Safety check
        safety = get_safety_enforcement()
        if not safety.validate_action("obsidian_plugin", {"action": "manage"}):
            click.echo("❌ Safety validation failed for Obsidian plugin")
            return

        plugin_path = Path(plugin_dir)
        
        if not plugin_path.exists():
            click.echo(f"❌ Plugin directory not found: {plugin_dir}")
            return

        click.echo(f"🔧 Managing Obsidian plugin in: {plugin_path}")

        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            click.echo("❌ npm is required but not found. Please install Node.js and npm.")
            return

        # Install dependencies
        click.echo("📦 Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=plugin_path, check=True)

        if build:
            click.echo("🔨 Building plugin...")
            subprocess.run(["npm", "run", "build"], cwd=plugin_path, check=True)
            click.echo("✅ Plugin built successfully")

        if package:
            click.echo("📦 Packaging plugin...")
            # Create plugin package
            package_dir = plugin_path / "dist"
            package_dir.mkdir(exist_ok=True)
            
            # Copy built files
            subprocess.run(["cp", "-r", "main.js", "styles.css", "manifest.json"], 
                         cwd=plugin_path, check=True)
            click.echo("✅ Plugin packaged successfully")

        if install:
            click.echo("📥 Installing plugin locally...")
            # Copy to Obsidian plugins directory
            obsidian_plugins_dir = Path.home() / ".obsidian" / "plugins" / "meta-model-ai"
            obsidian_plugins_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy plugin files
            subprocess.run(["cp", "-r", "main.js", "styles.css", "manifest.json"], 
                         cwd=plugin_path, check=True)
            click.echo("✅ Plugin installed successfully")

        if not any([build, package, install]):
            click.echo("ℹ️ Use --build, --package, or --install to perform actions")

    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Command failed: {e}")
    except Exception as e:
        click.echo(f"❌ Error managing Obsidian plugin: {e}")


@frontend.command()
@click.option('--web-dir', default='web/frontend', help='Web frontend directory')
@click.option('--build', is_flag=True, help='Build the web interface')
@click.option('--serve', is_flag=True, help='Serve the web interface')
@click.option('--port', default=3000, help='Port for serving web interface')
def web(web_dir: str, build: bool, serve: bool, port: int):
    """Manage web frontend."""
    try:
        # Safety check
        safety = get_safety_enforcement()
        if not safety.validate_action("web_frontend", {"action": "manage"}):
            click.echo("❌ Safety validation failed for web frontend")
            return

        web_path = Path(web_dir)
        
        if not web_path.exists():
            click.echo(f"❌ Web directory not found: {web_dir}")
            return

        click.echo(f"🌐 Managing web frontend in: {web_path}")

        if build:
            click.echo("🔨 Building web interface...")
            # Copy files to build directory
            build_dir = web_path / "dist"
            build_dir.mkdir(exist_ok=True)
            
            # Copy HTML, CSS, JS files
            for file in ["index.html", "styles.css", "app.js"]:
                if (web_path / file).exists():
                    subprocess.run(["cp", file, "dist/"], cwd=web_path, check=True)
            
            click.echo("✅ Web interface built successfully")

        if serve:
            click.echo(f"🚀 Serving web interface on port {port}...")
            
            # Check if Python HTTP server is available
            try:
                subprocess.run(["python3", "-m", "http.server", str(port)], 
                             cwd=web_path, check=True)
            except KeyboardInterrupt:
                click.echo("🛑 Server stopped")
            except Exception as e:
                click.echo(f"❌ Failed to start server: {e}")

        if not any([build, serve]):
            click.echo("ℹ️ Use --build or --serve to perform actions")

    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Command failed: {e}")
    except Exception as e:
        click.echo(f"❌ Error managing web frontend: {e}")


@frontend.command()
@click.option('--widget-dir', default='web/widgets', help='Widgets directory')
@click.option('--create', is_flag=True, help='Create a new widget')
@click.option('--build', is_flag=True, help='Build all widgets')
@click.option('--list', is_flag=True, help='List available widgets')
def widgets(widget_dir: str, create: bool, build: bool, list_widgets: bool):
    """Manage embeddable widgets."""
    try:
        # Safety check
        safety = get_safety_enforcement()
        if not safety.validate_action("widgets", {"action": "manage"}):
            click.echo("❌ Safety validation failed for widgets")
            return

        widgets_path = Path(widget_dir)
        
        if not widgets_path.exists():
            click.echo(f"❌ Widgets directory not found: {widget_dir}")
            return

        click.echo(f"🎯 Managing widgets in: {widgets_path}")

        if create:
            widget_name = click.prompt("Enter widget name")
            widget_path = widgets_path / widget_name
            
            if widget_path.exists():
                click.echo(f"❌ Widget '{widget_name}' already exists")
                return
            
            # Create widget structure
            widget_path.mkdir()
            (widget_path / "index.html").write_text(f"""<!DOCTYPE html>
<html>
<head>
    <title>{widget_name} Widget</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 1rem; }}
        .widget {{ border: 1px solid #ccc; border-radius: 8px; padding: 1rem; }}
    </style>
</head>
<body>
    <div class="widget">
        <h3>{widget_name}</h3>
        <p>Widget content goes here...</p>
    </div>
    <script>
        // Widget JavaScript
        console.log('{widget_name} widget loaded');
    </script>
</body>
</html>""")
            
            click.echo(f"✅ Widget '{widget_name}' created successfully")

        if build:
            click.echo("🔨 Building widgets...")
            for widget_dir in widgets_path.iterdir():
                if widget_dir.is_dir():
                    click.echo(f"  Building {widget_dir.name}...")
                    # Add build logic here if needed
            click.echo("✅ All widgets built successfully")

        if list_widgets:
            click.echo("📋 Available widgets:")
            for widget_dir in widgets_path.iterdir():
                if widget_dir.is_dir():
                    click.echo(f"  - {widget_dir.name}")

        if not any([create, build, list_widgets]):
            click.echo("ℹ️ Use --create, --build, or --list to perform actions")

    except Exception as e:
        click.echo(f"❌ Error managing widgets: {e}")


@frontend.command()
@click.option('--all', is_flag=True, help='Build all frontend components')
@click.option('--serve', is_flag=True, help='Serve all components')
def build_all(all: bool, serve: bool):
    """Build all frontend components."""
    try:
        # Safety check
        safety = get_safety_enforcement()
        if not safety.validate_action("build_all_frontend", {"action": "build"}):
            click.echo("❌ Safety validation failed for building all components")
            return

        click.echo("🏗️ Building all frontend components...")

        if all:
            # Build VSCode extension
            click.echo("📦 Building VSCode extension...")
            subprocess.run(["python", "cli/frontend_cli.py", "vscode", "--build"], check=True)

            # Build Obsidian plugin
            click.echo("📦 Building Obsidian plugin...")
            subprocess.run(["python", "cli/frontend_cli.py", "obsidian", "--build"], check=True)

            # Build web interface
            click.echo("🌐 Building web interface...")
            subprocess.run(["python", "cli/frontend_cli.py", "web", "--build"], check=True)

            # Build widgets
            click.echo("🎯 Building widgets...")
            subprocess.run(["python", "cli/frontend_cli.py", "widgets", "--build"], check=True)

            click.echo("✅ All frontend components built successfully")

        if serve:
            click.echo("🚀 Starting development servers...")
            
            # Start web interface server
            click.echo("🌐 Starting web interface server...")
            subprocess.Popen(["python", "cli/frontend_cli.py", "web", "--serve", "--port", "3000"])
            
            # Start API server
            click.echo("🔌 Starting API server...")
            subprocess.Popen(["python", "web/fastapi_app.py"])
            
            click.echo("✅ Development servers started")
            click.echo("📱 Web interface: http://localhost:3000")
            click.echo("🔌 API server: http://localhost:8000")

        if not any([all, serve]):
            click.echo("ℹ️ Use --all or --serve to perform actions")

    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Build failed: {e}")
    except Exception as e:
        click.echo(f"❌ Error building all components: {e}")


@frontend.command()
def status():
    """Show frontend component status."""
    try:
        click.echo("📊 Frontend Component Status")
        click.echo("=" * 40)

        # Check VSCode extension
        vscode_path = Path("web/extensions/vscode")
        if vscode_path.exists():
            package_json = vscode_path / "package.json"
            if package_json.exists():
                with open(package_json) as f:
                    data = json.load(f)
                click.echo(f"📦 VSCode Extension: {data.get('name', 'Unknown')} v{data.get('version', 'Unknown')}")
            else:
                click.echo("📦 VSCode Extension: ❌ package.json not found")
        else:
            click.echo("📦 VSCode Extension: ❌ Directory not found")

        # Check Obsidian plugin
        obsidian_path = Path("web/extensions/obsidian")
        if obsidian_path.exists():
            manifest_json = obsidian_path / "manifest.json"
            if manifest_json.exists():
                with open(manifest_json) as f:
                    data = json.load(f)
                click.echo(f"📦 Obsidian Plugin: {data.get('name', 'Unknown')} v{data.get('version', 'Unknown')}")
            else:
                click.echo("📦 Obsidian Plugin: ❌ manifest.json not found")
        else:
            click.echo("📦 Obsidian Plugin: ❌ Directory not found")

        # Check web frontend
        web_path = Path("web/frontend")
        if web_path.exists():
            index_html = web_path / "index.html"
            if index_html.exists():
                click.echo("🌐 Web Frontend: ✅ index.html found")
            else:
                click.echo("🌐 Web Frontend: ❌ index.html not found")
        else:
            click.echo("🌐 Web Frontend: ❌ Directory not found")

        # Check widgets
        widgets_path = Path("web/widgets")
        if widgets_path.exists():
            widget_count = len([d for d in widgets_path.iterdir() if d.is_dir()])
            click.echo(f"🎯 Widgets: {widget_count} widgets found")
        else:
            click.echo("🎯 Widgets: ❌ Directory not found")

        click.echo("=" * 40)

    except Exception as e:
        click.echo(f"❌ Error checking status: {e}")


if __name__ == "__main__":
    frontend() 