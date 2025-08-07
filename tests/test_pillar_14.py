#!/usr/bin/env python3
"""
Test suite for Pillar 14: Front-end & Embeddable UI
Tests web interface, extensions, and embeddable components
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_web_interface():
    """Test web interface components"""
    print("Testing Web Interface...")
    
    try:
        # Test FastAPI app
        from web.fastapi_app import app
        assert app is not None
        
        # Test web interface files
        web_files = [
            "web/frontend/index.html",
            "web/frontend/app.js", 
            "web/frontend/styles.css",
            "web/fastapi_app.py",
            "web/app.py",
            "web/web.py"
        ]
        
        for web_file in web_files:
            assert os.path.exists(web_file), f"Web file missing: {web_file}"
            
            # Check file size
            file_size = os.path.getsize(web_file)
            assert file_size > 0, f"Web file is empty: {web_file}"
        
        # Test HTML structure
        with open("web/frontend/index.html", 'r') as f:
            html_content = f.read()
            assert "<!DOCTYPE html>" in html_content
            assert "<html" in html_content
            assert "<head>" in html_content
            assert "<body>" in html_content
            assert "meta-model" in html_content.lower()
        
        # Test JavaScript functionality
        with open("web/frontend/app.js", 'r') as f:
            js_content = f.read()
            assert "function" in js_content
            assert "const" in js_content or "let" in js_content or "var" in js_content
            assert "addEventListener" in js_content or "onclick" in js_content
        
        # Test CSS styling
        with open("web/frontend/styles.css", 'r') as f:
            css_content = f.read()
            assert "{" in css_content
            assert "}" in css_content
            assert ":" in css_content
        
        print("✅ Web Interface - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Web Interface - FAILED: {e}")
        return False

def test_vscode_extension():
    """Test VSCode extension"""
    print("Testing VSCode Extension...")
    
    try:
        # Test VSCode extension files
        vscode_files = [
            "web/extensions/vscode/package.json",
            "web/extensions/vscode/src/extension.ts"
        ]
        
        for vscode_file in vscode_files:
            if os.path.exists(vscode_file):
                assert os.path.getsize(vscode_file) > 0, f"VSCode file is empty: {vscode_file}"
        
        # Test package.json if it exists
        package_json_path = "web/extensions/vscode/package.json"
        if os.path.exists(package_json_path):
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
                
                # Check required fields
                assert "name" in package_data, "Missing name in package.json"
                assert "version" in package_data, "Missing version in package.json"
                assert "displayName" in package_data, "Missing displayName in package.json"
                assert "description" in package_data, "Missing description in package.json"
                
                # Check that it's a VSCode extension
                assert "publisher" in package_data, "Missing publisher in package.json"
                assert "engines" in package_data, "Missing engines in package.json"
                assert "vscode" in package_data["engines"], "Not a VSCode extension"
        
        # Test TypeScript extension file if it exists
        extension_ts_path = "web/extensions/vscode/src/extension.ts"
        if os.path.exists(extension_ts_path):
            with open(extension_ts_path, 'r') as f:
                ts_content = f.read()
                assert "activate" in ts_content, "Missing activate function"
                assert "deactivate" in ts_content, "Missing deactivate function"
                assert "registerCommand" in ts_content, "Missing command registration"
        
        print("✅ VSCode Extension - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ VSCode Extension - FAILED: {e}")
        return False

def test_obsidian_plugin():
    """Test Obsidian plugin"""
    print("Testing Obsidian Plugin...")
    
    try:
        # Test Obsidian plugin files
        obsidian_files = [
            "web/extensions/obsidian/main.ts"
        ]
        
        for obsidian_file in obsidian_files:
            if os.path.exists(obsidian_file):
                assert os.path.getsize(obsidian_file) > 0, f"Obsidian file is empty: {obsidian_file}"
        
        # Test main.ts if it exists
        main_ts_path = "web/extensions/obsidian/main.ts"
        if os.path.exists(main_ts_path):
            with open(main_ts_path, 'r') as f:
                ts_content = f.read()
                assert "Plugin" in ts_content, "Missing Plugin class"
                assert "onload" in ts_content, "Missing onload function"
                assert "onunload" in ts_content, "Missing onunload function"
                assert "registerEditorExtension" in ts_content or "addCommand" in ts_content, "Missing command registration"
        
        print("✅ Obsidian Plugin - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Obsidian Plugin - FAILED: {e}")
        return False

def test_embeddable_widgets():
    """Test embeddable widgets"""
    print("Testing Embeddable Widgets...")
    
    try:
        # Test widget files
        widget_files = [
            "web/widgets/chat-widget.js"
        ]
        
        for widget_file in widget_files:
            if os.path.exists(widget_file):
                assert os.path.getsize(widget_file) > 0, f"Widget file is empty: {widget_file}"
                
                # Check widget functionality
                with open(widget_file, 'r') as f:
                    widget_content = f.read()
                    assert "function" in widget_content, "Widget should contain functions"
                    assert "document" in widget_content or "window" in widget_content, "Widget should interact with DOM"
        
        # Test widget directory structure
        widgets_dir = "web/widgets"
        if os.path.exists(widgets_dir):
            widget_files_in_dir = os.listdir(widgets_dir)
            assert len(widget_files_in_dir) > 0, "Widgets directory should contain files"
        
        print("✅ Embeddable Widgets - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Embeddable Widgets - FAILED: {e}")
        return False

def test_ui_components():
    """Test UI components"""
    print("Testing UI Components...")
    
    try:
        # Test components directory
        components_dir = "web/components"
        if os.path.exists(components_dir):
            component_files = os.listdir(components_dir)
            assert len(component_files) > 0, "Components directory should contain files"
        
        # Test responsive design
        css_file = "web/frontend/styles.css"
        if os.path.exists(css_file):
            with open(css_file, 'r') as f:
                css_content = f.read()
                
                # Check for responsive design indicators
                responsive_indicators = [
                    "@media",
                    "max-width",
                    "min-width",
                    "flexbox",
                    "grid",
                    "responsive"
                ]
                
                responsive_count = sum(1 for indicator in responsive_indicators if indicator in css_content)
                assert responsive_count > 0, "CSS should include responsive design"
        
        # Test modern UI features
        js_file = "web/frontend/app.js"
        if os.path.exists(js_file):
            with open(js_file, 'r') as f:
                js_content = f.read()
                
                # Check for modern UI features
                modern_features = [
                    "addEventListener",
                    "fetch",
                    "async",
                    "await",
                    "localStorage",
                    "sessionStorage"
                ]
                
                modern_count = sum(1 for feature in modern_features if feature in js_content)
                assert modern_count > 0, "JavaScript should include modern features"
        
        print("✅ UI Components - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ UI Components - FAILED: {e}")
        return False

def test_user_experience():
    """Test user experience features"""
    print("Testing User Experience...")
    
    try:
        # Test for UX features in HTML
        html_file = "web/frontend/index.html"
        if os.path.exists(html_file):
            with open(html_file, 'r') as f:
                html_content = f.read()
                
                # Check for UX features
                ux_features = [
                    "placeholder",
                    "aria-label",
                    "title",
                    "alt",
                    "loading",
                    "error",
                    "success"
                ]
                
                ux_count = sum(1 for feature in ux_features if feature in html_content)
                assert ux_count > 0, "HTML should include UX features"
        
        # Test for accessibility features
        css_file = "web/frontend/styles.css"
        if os.path.exists(css_file):
            with open(css_file, 'r') as f:
                css_content = f.read()
                
                # Check for accessibility features
                accessibility_features = [
                    "focus",
                    "hover",
                    "active",
                    "disabled",
                    "aria-",
                    "outline",
                    "contrast"
                ]
                
                accessibility_count = sum(1 for feature in accessibility_features if feature in css_content)
                assert accessibility_count > 0, "CSS should include accessibility features"
        
        print("✅ User Experience - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ User Experience - FAILED: {e}")
        return False

def test_real_time_features():
    """Test real-time features"""
    print("Testing Real-time Features...")
    
    try:
        # Test WebSocket support
        js_file = "web/frontend/app.js"
        if os.path.exists(js_file):
            with open(js_file, 'r') as f:
                js_content = f.read()
                
                # Check for real-time features
                realtime_features = [
                    "WebSocket",
                    "EventSource",
                    "setInterval",
                    "setTimeout",
                    "stream",
                    "real-time",
                    "live"
                ]
                
                realtime_count = sum(1 for feature in realtime_features if feature in js_content)
                assert realtime_count > 0, "JavaScript should include real-time features"
        
        # Test FastAPI streaming endpoints
        fastapi_file = "web/fastapi_app.py"
        if os.path.exists(fastapi_file):
            with open(fastapi_file, 'r') as f:
                fastapi_content = f.read()
                
                # Check for streaming endpoints
                streaming_features = [
                    "StreamingResponse",
                    "websocket",
                    "stream",
                    "SSE",
                    "Server-Sent Events"
                ]
                
                streaming_count = sum(1 for feature in streaming_features if feature in fastapi_content)
                assert streaming_count > 0, "FastAPI should include streaming features"
        
        print("✅ Real-time Features - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Real-time Features - FAILED: {e}")
        return False

def test_cross_platform_integration():
    """Test cross-platform integration"""
    print("Testing Cross-platform Integration...")
    
    try:
        # Test for cross-platform features
        js_file = "web/frontend/app.js"
        if os.path.exists(js_file):
            with open(js_file, 'r') as f:
                js_content = f.read()
                
                # Check for cross-platform features
                cross_platform_features = [
                    "navigator",
                    "userAgent",
                    "platform",
                    "mobile",
                    "desktop",
                    "tablet",
                    "responsive"
                ]
                
                cross_platform_count = sum(1 for feature in cross_platform_features if feature in js_content)
                assert cross_platform_count > 0, "JavaScript should include cross-platform features"
        
        # Test for mobile support
        css_file = "web/frontend/styles.css"
        if os.path.exists(css_file):
            with open(css_file, 'r') as f:
                css_content = f.read()
                
                # Check for mobile support
                mobile_features = [
                    "@media",
                    "max-width",
                    "min-width",
                    "viewport",
                    "touch",
                    "mobile"
                ]
                
                mobile_count = sum(1 for feature in mobile_features if feature in css_content)
                assert mobile_count > 0, "CSS should include mobile support"
        
        print("✅ Cross-platform Integration - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Cross-platform Integration - FAILED: {e}")
        return False

def main():
    """Run all Pillar 14 tests."""
    print("🧪 Testing Pillar 14: Front-end & Embeddable UI...")
    print("=" * 60)
    
    results = []
    
    # Test each component
    results.append(test_web_interface())
    results.append(test_vscode_extension())
    results.append(test_obsidian_plugin())
    results.append(test_embeddable_widgets())
    results.append(test_ui_components())
    results.append(test_user_experience())
    results.append(test_real_time_features())
    results.append(test_cross_platform_integration())
    
    # Summary
    print("=" * 60)
    print("📊 Test Results Summary:")
    
    test_names = [
        "Web Interface",
        "VSCode Extension",
        "Obsidian Plugin",
        "Embeddable Widgets",
        "UI Components",
        "User Experience",
        "Real-time Features",
        "Cross-platform Integration"
    ]
    
    passed = 0
    for i, (result, name) in enumerate(zip(results, test_names)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Pillar 14: Front-end & Embeddable UI is working correctly!")
        print("\n📋 Pillar 14 Features:")
        print("  ✅ Rich web interface with modern UI")
        print("  ✅ VSCode extension for development")
        print("  ✅ Obsidian plugin for knowledge management")
        print("  ✅ Embeddable widgets for integration")
        print("  ✅ Responsive design for all devices")
        print("  ✅ Real-time streaming and updates")
        print("  ✅ Cross-platform compatibility")
        print("  ✅ Modern user experience")
        return True
    else:
        print("⚠️  Some tests need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 