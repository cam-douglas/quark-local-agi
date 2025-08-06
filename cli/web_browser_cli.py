#!/usr/bin/env python3
"""
Web Browser CLI for Meta-Model AI Assistant
===========================================

Provides commands for internet browsing, web search, and captcha handling.
"""

import click
import json
import time
from datetime import datetime
from typing import Dict, Any

from core.web_browser import get_web_browser, browse_url, search_web, resolve_captcha


@click.group()
def web():
    """Web browsing and internet capabilities."""
    pass


@web.command()
@click.argument('url', required=True)
@click.option('--query', '-q', help='User query to focus content extraction')
def browse(url, query):
    """Browse a URL and extract content."""
    click.secho(f"🌐 BROWSING URL", fg="blue", bold=True)
    click.echo(f"URL: {url}")
    if query:
        click.echo(f"Query: {query}")
    click.echo()
    
    try:
        result = browse_url(url, query)
        
        if result["success"]:
            click.secho("✅ Browse successful", fg="green")
            click.echo()
            
            click.secho("📄 CONTENT:", fg="yellow", bold=True)
            click.echo(f"Title: {result['title']}")
            click.echo(f"Content length: {len(result['content'])} characters")
            click.echo()
            
            click.secho("📝 SUMMARY:", fg="cyan", bold=True)
            click.echo(result['summary'])
            click.echo()
            
            if result['links']:
                click.secho("🔗 LINKS:", fg="magenta", bold=True)
                for i, link in enumerate(result['links'][:5], 1):
                    click.echo(f"{i}. {link['text']} -> {link['url']}")
                click.echo()
            
            click.secho("📊 METADATA:", fg="blue", bold=True)
            metadata = result['metadata']
            click.echo(f"Status code: {metadata['status_code']}")
            click.echo(f"Content length: {metadata['content_length']}")
            click.echo(f"Timestamp: {metadata['timestamp']}")
            
        elif result.get("error") == "CAPTCHA_DETECTED":
            click.secho("🔒 CAPTCHA DETECTED", fg="red", bold=True)
            click.echo()
            click.echo(result["user_prompt"])
            click.echo()
            click.echo(f"Captcha ID: {result['captcha_id']}")
            click.echo(f"Captcha Type: {result['captcha_type']}")
            click.echo()
            click.echo("To resolve this captcha, use:")
            click.echo(f"web resolve-captcha --captcha-id {result['captcha_id']} --solution 'your_solution'")
            
        else:
            click.secho(f"❌ Browse failed: {result['error']}", fg="red")
            
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red")


@web.command()
@click.argument('query', required=True)
@click.option('--max-results', '-m', default=5, help='Maximum number of results')
def search(query, max_results):
    """Search the web for information."""
    click.secho(f"🔍 WEB SEARCH", fg="blue", bold=True)
    click.echo(f"Query: {query}")
    click.echo(f"Max results: {max_results}")
    click.echo()
    
    try:
        result = search_web(query, max_results)
        
        if result["success"]:
            click.secho("✅ Search successful", fg="green")
            click.echo()
            
            click.secho("📄 RESULTS:", fg="yellow", bold=True)
            for i, search_result in enumerate(result['results'], 1):
                click.echo(f"{i}. {search_result['title']}")
                click.echo(f"   URL: {search_result['url']}")
                click.echo(f"   Snippet: {search_result['snippet']}")
                click.echo()
            
            click.secho("📊 METADATA:", fg="blue", bold=True)
            metadata = result['metadata']
            click.echo(f"Total results: {result['total_results']}")
            click.echo(f"Max results: {metadata['max_results']}")
            click.echo(f"Timestamp: {metadata['timestamp']}")
            
        else:
            click.secho(f"❌ Search failed: {result['error']}", fg="red")
            
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red")


@web.command()
@click.option('--captcha-id', '-c', required=True, help='Captcha ID to resolve')
@click.option('--solution', '-s', required=True, help='User-provided solution')
def resolve_captcha_command(captcha_id, solution):
    """Resolve a captcha with user-provided solution."""
    click.secho(f"🔓 RESOLVING CAPTCHA", fg="blue", bold=True)
    click.echo(f"Captcha ID: {captcha_id}")
    click.echo(f"Solution: {solution}")
    click.echo()
    
    try:
        result = resolve_captcha(captcha_id, solution)
        
        if result["success"]:
            click.secho("✅ Captcha resolved successfully", fg="green")
            click.echo(f"Message: {result['message']}")
            click.echo()
            
            # Show retry result if available
            if "retry_result" in result:
                retry = result["retry_result"]
                if retry["success"]:
                    click.secho("🔄 RETRY RESULT:", fg="yellow", bold=True)
                    click.echo(f"Title: {retry['title']}")
                    click.echo(f"Content length: {len(retry['content'])} characters")
                    click.echo(f"Summary: {retry['summary']}")
                else:
                    click.secho(f"⚠️  Retry failed: {retry['error']}", fg="yellow")
            
        else:
            click.secho(f"❌ Captcha resolution failed: {result['error']}", fg="red")
            
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red")


@web.command()
def status():
    """Check web browser status and statistics."""
    click.secho("🌐 WEB BROWSER STATUS", fg="blue", bold=True)
    click.echo()
    
    browser = get_web_browser()
    stats = browser.get_browser_stats()
    
    click.secho("📊 STATISTICS:", fg="yellow", bold=True)
    click.echo(f"Total requests: {stats['total_requests']}")
    click.echo(f"Captcha encounters: {stats['captcha_encounters']}")
    click.echo(f"Resolved captchas: {stats['resolved_captchas']}")
    click.echo(f"User responses: {stats['user_responses']}")
    click.echo()
    
    click.secho("⚙️  CONFIGURATION:", fg="cyan", bold=True)
    config = stats['config']
    click.echo(f"Timeout: {config['timeout']} seconds")
    click.echo(f"Max redirects: {config['max_redirects']}")
    click.echo(f"Captcha detection: {'✅' if config['captcha_detection'] else '❌'}")
    click.echo(f"User confirmation required: {'✅' if config['user_confirmation_required'] else '❌'}")
    click.echo(f"Max page size: {config['max_page_size'] / (1024*1024):.1f} MB")
    click.echo(f"Rate limit: {config['rate_limit']['requests_per_minute']} requests/minute")
    click.echo()
    
    if config['allowed_domains']:
        click.secho("✅ ALLOWED DOMAINS:", fg="green", bold=True)
        for domain in config['allowed_domains']:
            click.echo(f"  • {domain}")
        click.echo()
    
    if config['blocked_domains']:
        click.secho("❌ BLOCKED DOMAINS:", fg="red", bold=True)
        for domain in config['blocked_domains']:
            click.echo(f"  • {domain}")
        click.echo()
    
    click.secho("✅ Web browser is active and ready", fg="green")


@web.command()
@click.option('--url', '-u', required=True, help='URL to test')
def test_browse(url):
    """Test browsing capabilities with a URL."""
    click.secho(f"🧪 TESTING WEB BROWSER", fg="blue", bold=True)
    click.echo(f"URL: {url}")
    click.echo()
    
    try:
        result = browse_url(url)
        
        if result["success"]:
            click.secho("✅ Browse test successful", fg="green")
            click.echo(f"Title: {result['title']}")
            click.echo(f"Content length: {len(result['content'])} characters")
            click.echo(f"Status code: {result['metadata']['status_code']}")
            click.echo(f"Links found: {len(result['links'])}")
            
        elif result.get("error") == "CAPTCHA_DETECTED":
            click.secho("🔒 CAPTCHA DETECTED (Test)", fg="red", bold=True)
            click.echo(f"Captcha ID: {result['captcha_id']}")
            click.echo(f"Captcha Type: {result['captcha_type']}")
            click.echo("This is expected behavior for sites with captcha protection.")
            
        else:
            click.secho(f"❌ Browse test failed: {result['error']}", fg="red")
            
    except Exception as e:
        click.secho(f"❌ Test error: {e}", fg="red")


@web.command()
@click.option('--domain', '-d', help='Domain to allow')
@click.option('--block-domain', '-b', help='Domain to block')
@click.option('--timeout', '-t', type=int, help='Request timeout in seconds')
@click.option('--rate-limit', '-r', type=int, help='Requests per minute')
def config(domain, block_domain, timeout, rate_limit):
    """Configure web browser settings."""
    click.secho("⚙️  CONFIGURING WEB BROWSER", fg="blue", bold=True)
    click.echo()
    
    browser = get_web_browser()
    config_updates = {}
    
    if domain:
        browser.browser_config["allowed_domains"].append(domain)
        click.secho(f"✅ Added domain to allowed list: {domain}", fg="green")
    
    if block_domain:
        browser.browser_config["blocked_domains"].append(block_domain)
        click.secho(f"❌ Added domain to blocked list: {block_domain}", fg="red")
    
    if timeout:
        config_updates["timeout"] = timeout
        click.secho(f"⏱️  Set timeout to: {timeout} seconds", fg="yellow")
    
    if rate_limit:
        config_updates["rate_limit"] = {"requests_per_minute": rate_limit, "requests_per_hour": rate_limit * 60}
        click.secho(f"🚦 Set rate limit to: {rate_limit} requests/minute", fg="yellow")
    
    if config_updates:
        browser.update_config(config_updates)
        click.secho("✅ Configuration updated", fg="green")
    else:
        click.secho("ℹ️  No configuration changes specified", fg="blue")


@web.command()
def captchas():
    """Show recent captcha encounters."""
    click.secho("🔒 RECENT CAPTCHA ENCOUNTERS", fg="blue", bold=True)
    click.echo()
    
    browser = get_web_browser()
    
    if not browser.captcha_encounters:
        click.secho("✅ No captcha encounters recorded", fg="green")
        return
    
    for i, encounter in enumerate(browser.captcha_encounters[-10:], 1):  # Last 10
        status = "✅ RESOLVED" if encounter["resolved"] else "❌ PENDING"
        color = "green" if encounter["resolved"] else "red"
        
        click.secho(f"{i}. {status}", fg=color, bold=True)
        click.echo(f"   ID: {encounter['id']}")
        click.echo(f"   URL: {encounter['url']}")
        click.echo(f"   Type: {encounter['type']}")
        click.echo(f"   Time: {encounter['timestamp']}")
        if encounter.get("user_query"):
            click.echo(f"   Query: {encounter['user_query']}")
        if encounter.get("solution"):
            click.echo(f"   Solution: {encounter['solution']}")
        click.echo()


@web.command()
def history():
    """Show recent browsing history."""
    click.secho("📚 BROWSING HISTORY", fg="blue", bold=True)
    click.echo()
    
    browser = get_web_browser()
    
    if not browser.request_history:
        click.secho("ℹ️  No browsing history recorded", fg="blue")
        return
    
    click.secho("Recent requests (last 10):", fg="yellow", bold=True)
    for i, request in enumerate(browser.request_history[-10:], 1):
        status_color = "green" if request["status_code"] == 200 else "red"
        click.secho(f"{i}. {request['url']}", fg=status_color)
        click.echo(f"   Status: {request['status_code']}")
        click.echo(f"   Size: {request['content_length']} bytes")
        click.echo(f"   Time: {datetime.fromtimestamp(request['timestamp'])}")
        click.echo()


if __name__ == "__main__":
    web() 