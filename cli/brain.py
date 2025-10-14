#!/usr/bin/env python3
"""
Second Brain CLI - Your personal semantic memory assistant

Commands:
  brain add <text>         Add a new memory
  brain recall <query>     Search for memories
  brain list [query]       List all memories (or search)
  brain delete <id>        Delete a specific memory
  brain clear              Delete all memories
  brain stats              Show statistics

Examples:
  brain add "passport in blue suitcase"
  brain recall "where's my passport?"
  brain list
  brain delete a3f2
"""

import sys
import click
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(__file__ + "/../.."))

from core.memory_manager import get_memory_manager
import config


# =============================================================================
# CLI STYLING
# =============================================================================

def success(text: str) -> str:
    """Format success message in green."""
    return click.style(text, fg='green', bold=True)

def error(text: str) -> str:
    """Format error message in red."""
    return click.style(text, fg='red', bold=True)

def info(text: str) -> str:
    """Format info message in blue."""
    return click.style(text, fg='blue')

def dim(text: str) -> str:
    """Format secondary text in gray."""
    return click.style(text, fg='bright_black')

def bold(text: str) -> str:
    """Format text in bold."""
    return click.style(text, bold=True)

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        # Show relative time if recent, otherwise date
        now = datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            # Today - show time
            if diff.seconds < 60:
                return "just now"
            elif diff.seconds < 3600:
                mins = diff.seconds // 60
                return f"{mins} min ago"
            else:
                hours = diff.seconds // 3600
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.days == 1:
            return "yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        else:
            return dt.strftime("%b %d, %Y")
    except:
        return timestamp_str


# =============================================================================
# CLI GROUP
# =============================================================================

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """
    🧠 Second Brain - Your personal semantic memory assistant
    
    Store memories and recall them later using natural language.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# =============================================================================
# ADD COMMAND
# =============================================================================

@cli.command()
@click.argument('text', nargs=-1, required=True)
@click.option('--no-confirm', is_flag=True, help='Skip confirmation step')
def add(text, no_confirm):
    """
    Add a new memory to your brain.
    
    Example:
        brain add "My passport is in the blue suitcase"
    """
    memory_text = ' '.join(text)
    
    if not memory_text.strip():
        click.echo(error("❌ Error: Memory text cannot be empty"))
        return
    
    # Show preview and confirm (unless --no-confirm)
    if config.CONFIRM_BEFORE_SAVE and not no_confirm:
        click.echo("\n" + "─" * 60)
        click.echo(bold("💾 Memory ready to save:"))
        click.echo(f"\n{memory_text}\n")
        click.echo(dim(f"Timestamp: {datetime.now().strftime('%b %d, %Y at %I:%M %p')}"))
        click.echo("─" * 60)
        
        if not click.confirm(success("Save this memory?"), default=True):
            click.echo(info("✓ Cancelled"))
            return
    
    # Add the memory
    try:
        manager = get_memory_manager()
        result = manager.add_memory(memory_text)
        
        click.echo(success(f"\n✓ Memory saved!") + dim(f" [ID: {result['id']}]"))
        
    except Exception as e:
        click.echo(error(f"❌ Error adding memory: {e}"))
        sys.exit(1)


# =============================================================================
# RECALL COMMAND
# =============================================================================

@cli.command()
@click.argument('query', nargs=-1, required=True)
@click.option('--debug', is_flag=True, help='Show scoring details')
@click.option('--limit', '-n', type=int, default=config.TOP_N_RESULTS, 
              help=f'Number of results (default: {config.TOP_N_RESULTS})')
@click.option('--threshold', '-t', type=float, default=config.CONFIDENCE_THRESHOLD,
              help=f'Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})')
def recall(query, debug, limit, threshold):
    """
    Search your memories using natural language.
    
    Example:
        brain recall "where did I put my passport?"
    """
    query_text = ' '.join(query)
    
    if not query_text.strip():
        click.echo(error("❌ Error: Query cannot be empty"))
        return
    
    try:
        manager = get_memory_manager()
        
        # Check if any memories exist
        if manager.count_memories() == 0:
            click.echo(info("🤷 Your brain is empty! Add some memories first."))
            click.echo(dim("\nTry: brain add \"your memory here\""))
            return
        
        # Search for memories
        results = manager.recall_memory(
            query=query_text,
            n_results=limit,
            min_confidence=threshold,
            debug=debug
        )
        
        if not results:
            click.echo(info(f"🤷 No confident matches found for: \"{query_text}\""))
            click.echo(dim(f"\n(Try lowering threshold with --threshold or use 'brain list' to see all memories)"))
            return
        
        # Display results
        click.echo(f"\n{bold('🔍 Found')} {len(results)} {bold('matching memories:')}\n")
        
        for i, result in enumerate(results, 1):
            # Memory text
            click.echo(f"{bold(f'{i}.')} {result['text']}")
            
            # Metadata line
            meta_parts = [
                dim(f"[{result['id']}]"),
                dim(format_timestamp(result['timestamp']))
            ]
            
            # Score
            score_color = 'green' if result['score'] > 0.7 else 'yellow' if result['score'] > 0.5 else 'red'
            meta_parts.append(click.style(f"Score: {result['score']:.2f}", fg=score_color))
            
            click.echo("   " + " · ".join(meta_parts))
            
            # Debug info
            if debug and 'debug' in result:
                d = result['debug']
                click.echo(dim(f"   Debug: similarity={d['similarity_score']:.3f}, "
                             f"recency={d['recency_score']:.3f}, "
                             f"distance={d['distance']:.3f}"))
            
            click.echo()  # Blank line between results
        
    except Exception as e:
        click.echo(error(f"❌ Error recalling memory: {e}"))
        import traceback
        if debug:
            traceback.print_exc()
        sys.exit(1)


# =============================================================================
# LIST COMMAND
# =============================================================================

@cli.command()
@click.argument('query', nargs=-1, required=False)
@click.option('--limit', '-n', type=int, help='Maximum number of results')
def list(query, limit):
    """
    List all memories, optionally filtered by a search query.
    
    Examples:
        brain list                    # Show all memories
        brain list "passport"         # Search for passport-related memories
        brain list --limit 10         # Show only 10 most recent
    """
    try:
        manager = get_memory_manager()
        
        # Check if any memories exist
        total = manager.count_memories()
        if total == 0:
            click.echo(info("🤷 Your brain is empty! Add some memories first."))
            click.echo(dim("\nTry: brain add \"your memory here\""))
            return
        
        # Get memories
        query_text = ' '.join(query) if query else None
        memories = manager.list_memories(query=query_text, limit=limit)
        
        if not memories:
            click.echo(info(f"🤷 No memories found matching: \"{query_text}\""))
            return
        
        # Display header
        if query_text:
            click.echo(f"\n{bold('🔍 Memories matching')} \"{query_text}\" {dim(f'({len(memories)} results)')}\n")
        else:
            click.echo(f"\n{bold('📚 All memories')} {dim(f'({len(memories)} of {total})')}\n")
        
        # Display memories
        for i, memory in enumerate(memories, 1):
            # Memory text (truncate if too long)
            text = memory['text']
            if len(text) > config.PREVIEW_MAX_LENGTH:
                text = text[:config.PREVIEW_MAX_LENGTH] + "..."
            
            click.echo(f"{bold(f'{i}.')} {text}")
            
            # Metadata
            meta_parts = [
                dim(f"[{memory['id']}]"),
                dim(format_timestamp(memory['timestamp']))
            ]
            
            # Score if from search
            if 'score' in memory:
                score_color = 'green' if memory['score'] > 0.7 else 'yellow'
                meta_parts.append(click.style(f"{memory['score']:.2f}", fg=score_color))
            
            click.echo("   " + " · ".join(meta_parts))
            click.echo()  # Blank line
        
    except Exception as e:
        click.echo(error(f"❌ Error listing memories: {e}"))
        sys.exit(1)


# =============================================================================
# DELETE COMMAND
# =============================================================================

@cli.command()
@click.argument('memory_id')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
def delete(memory_id, yes):
    """
    Delete a specific memory by ID.
    
    Example:
        brain delete a3f2
    """
    try:
        manager = get_memory_manager()
        
        # Get the memory to show what will be deleted
        memory = manager.get_memory(memory_id)
        
        if not memory:
            click.echo(error(f"❌ Memory not found: {memory_id}"))
            click.echo(dim("\nUse 'brain list' to see all memory IDs"))
            return
        
        # Show what will be deleted
        click.echo(f"\n{bold('⚠️  About to delete:')}")
        click.echo(f"\n[{memory['id']}] {memory['text']}")
        click.echo(dim(f"Added: {format_timestamp(memory['timestamp'])}"))
        click.echo()
        
        # Confirm deletion
        if not yes:
            if not click.confirm(error("Delete this memory?"), default=False):
                click.echo(info("✓ Cancelled"))
                return
        
        # Delete
        if manager.delete_memory(memory_id):
            click.echo(success("✓ Memory deleted"))
        else:
            click.echo(error(f"❌ Failed to delete memory: {memory_id}"))
            sys.exit(1)
        
    except Exception as e:
        click.echo(error(f"❌ Error deleting memory: {e}"))
        sys.exit(1)


# =============================================================================
# CLEAR COMMAND
# =============================================================================

@cli.command()
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
def clear(yes):
    """
    Delete ALL memories from your brain.
    
    ⚠️  WARNING: This cannot be undone!
    """
    try:
        manager = get_memory_manager()
        
        total = manager.count_memories()
        
        if total == 0:
            click.echo(info("Your brain is already empty!"))
            return
        
        # Confirm deletion
        click.echo(error(f"\n⚠️  WARNING: This will delete ALL {total} memories!"))
        click.echo(error("This action CANNOT be undone.\n"))
        
        if not yes:
            if not click.confirm(error("Are you absolutely sure?"), default=False):
                click.echo(info("✓ Cancelled"))
                return
            
            # Double confirmation for safety
            if not click.confirm(error("Really delete everything?"), default=False):
                click.echo(info("✓ Cancelled"))
                return
        
        # Clear all
        manager.clear_all_memories()
        click.echo(success(f"✓ Deleted all {total} memories"))
        
    except Exception as e:
        click.echo(error(f"❌ Error clearing memories: {e}"))
        sys.exit(1)


# =============================================================================
# STATS COMMAND
# =============================================================================

@cli.command()
def stats():
    """
    Show statistics about your memory storage.
    """
    try:
        manager = get_memory_manager()
        
        total = manager.count_memories()
        
        click.echo(f"\n{bold('📊 Second Brain Statistics')}\n")
        click.echo(f"Total memories: {bold(str(total))}")
        
        if total > 0:
            # Get most recent memory
            recent = manager.list_memories(limit=1)
            if recent:
                click.echo(f"Latest memory: {dim(format_timestamp(recent[0]['timestamp']))}")
        
        click.echo(f"\nStorage location: {dim(str(config.DATA_DIR))}")
        click.echo(f"Embedding model: {dim(config.EMBEDDING_MODEL.split('/')[-1])}")
        click.echo(f"Embedding dimension: {dim(str(manager.embedding_engine.dimension))}")
        
        # Configuration
        click.echo(f"\n{bold('⚙️  Configuration')}\n")
        click.echo(f"Similarity weight: {config.SIMILARITY_WEIGHT}")
        click.echo(f"Recency weight: {config.RECENCY_WEIGHT}")
        click.echo(f"Recency decay rate: {config.RECENCY_DECAY_RATE}")
        click.echo(f"Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
        click.echo(f"Top results: {config.TOP_N_RESULTS}")
        click.echo()
        
    except Exception as e:
        click.echo(error(f"❌ Error getting stats: {e}"))
        sys.exit(1)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    cli()

