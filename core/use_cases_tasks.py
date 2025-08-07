# mapping of pillars → sub-tasks
# Based on the 21-pillar development roadmap for Quark AI Assistant

PILLARS = {
    "Natural Language Understanding": [
        "Intent classification",
        "Entity recognition", 
        "Sentiment analysis",
        "Syntax parsing",
        "Semantic understanding",
        "Context awareness"
    ],
    "Knowledge Retrieval": [
        "Keyword-based document search",
        "Vector-based semantic retrieval",
        "FAQ lookup",
        "Knowledge graph queries",
        "Cross-document reasoning",
        "Long-term memory access"
    ],
    "Reasoning": [
        "Chain-of-thought reasoning",
        "Deduction",
        "Induction", 
        "Abductive reasoning",
        "Causal reasoning",
        "Temporal reasoning",
        "Multi-step problem solving"
    ],
    "Planning": [
        "Task decomposition",
        "Sequence planning",
        "Resource allocation",
        "Goal setting",
        "Constraint satisfaction",
        "Multi-scale planning"
    ],
    "Memory & Context": [
        "Short-term context management",
        "Long-term memory storage",
        "Memory eviction",
        "Context window management",
        "Knowledge persistence",
        "Memory consolidation"
    ],
    "Metrics & Evaluation": [
        "Performance monitoring",
        "Error tracking",
        "Latency measurement",
        "Accuracy assessment",
        "User satisfaction metrics",
        "Model performance analytics"
    ],
    "Self-Improvement": [
        "Automated fine-tuning",
        "Online learning",
        "Model upgrade management",
        "Performance optimization",
        "Capability bootstrapping",
        "Self-reflection loops"
    ],
    "Streaming & Real-Time": [
        "Token streaming",
        "Real-time response generation",
        "WebSocket communication",
        "FastAPI endpoints",
        "Live model inference",
        "Progressive output"
    ],
    "Testing & Quality": [
        "Unit testing",
        "Integration testing",
        "End-to-end testing",
        "Performance benchmarking",
        "Security testing",
        "Adversarial testing"
    ],
    "Deployment & Scaling": [
        "Container orchestration",
        "Load balancing",
        "Auto-scaling",
        "CI/CD pipeline",
        "Monitoring & alerting",
        "Disaster recovery"
    ],
    "Async & Parallel": [
        "Asynchronous execution",
        "Parallel model inference",
        "Concurrent task processing",
        "Multi-agent coordination",
        "Resource optimization",
        "Performance scaling"
    ],
    "Front-end & UI": [
        "Web interface development",
        "VSCode extension",
        "Obsidian plugin",
        "Embeddable widgets",
        "Rich UI components",
        "User experience design"
    ],
    "Safety & Alignment": [
        "Content filtering",
        "Safety guardrails",
        "RLHF integration",
        "Ethical AI practices",
        "Bias detection",
        "Harm prevention"
    ],
    "Meta-Learning": [
        "Self-monitoring agents",
        "Performance introspection",
        "Pipeline reconfiguration",
        "Meta-learning capabilities",
        "Self-improvement loops",
        "Adaptive behavior"
    ],
    "Knowledge Graphs": [
        "Knowledge graph construction",
        "Graph-based reasoning",
        "Cross-document linking",
        "Semantic knowledge representation",
        "Graph traversal",
        "Knowledge integration"
    ],
    "Generalized Reasoning": [
        "Simulation environments",
        "Multi-scale planning",
        "Resource management",
        "Temporal reasoning",
        "Complex goal decomposition",
        "Abstract thinking"
    ],
    "Social Intelligence": [
        "Theory of mind modeling",
        "Multi-agent collaboration",
        "Negotiation capabilities",
        "Social intelligence",
        "Belief modeling",
        "Collaborative problem solving"
    ],
    "Autonomous Goals": [
        "Autonomous goal setting",
        "Self-motivation systems",
        "Capability bootstrapping",
        "Objective identification",
        "Self-directed learning",
        "Intrinsic motivation"
    ],
    "Governance & Ethics": [
        "Ethics framework",
        "Human oversight systems",
        "Audit trails",
        "Governance protocols",
        "Safe deployment practices",
        "Transparency mechanisms"
    ]
}

# Development phases mapping
DEVELOPMENT_PHASES = {
    "Phase 1: Foundation": [
        "Natural Language Understanding",
        "Knowledge Retrieval", 
        "Reasoning",
        "Planning"
    ],
    "Phase 2: Core Framework": [
        "Memory & Context",
        "Metrics & Evaluation",
        "Self-Improvement"
    ],
    "Phase 3: Advanced Features": [
        "Streaming & Real-Time",
        "Testing & Quality",
        "Deployment & Scaling"
    ],
    "Phase 4: Intelligence Enhancement": [
        "Async & Parallel",
        "Front-end & UI",
        "Safety & Alignment",
        "Meta-Learning"
    ],
    "Phase 5: AGI Capabilities": [
        "Knowledge Graphs",
        "Generalized Reasoning",
        "Social Intelligence",
        "Autonomous Goals",
        "Governance & Ethics"
    ]
}

# Task complexity levels
TASK_COMPLEXITY = {
    "LOW": [
        "Intent classification",
        "Keyword-based document search",
        "Basic sentiment analysis",
        "Simple task decomposition"
    ],
    "MEDIUM": [
        "Entity recognition",
        "Vector-based semantic retrieval",
        "Chain-of-thought reasoning",
        "Resource allocation",
        "Performance monitoring"
    ],
    "HIGH": [
        "Cross-document reasoning",
        "Multi-step problem solving",
        "Multi-scale planning",
        "Knowledge graph construction",
        "Theory of mind modeling"
    ],
    "EXPERT": [
        "Autonomous goal setting",
        "Self-motivation systems",
        "Governance protocols",
        "Ethics framework",
        "Safe AGI deployment"
    ]
}

def list_categories():
    """Flatten all pillar sub-categories into a single list."""
    return [c for cats in PILLARS.values() for c in cats]

def get_pillar_for_task(task):
    """Find which pillar a specific task belongs to."""
    for pillar, tasks in PILLARS.items():
        if task in tasks:
            return pillar
    return None

def get_phase_for_pillar(pillar):
    """Find which development phase a pillar belongs to."""
    for phase, pillars in DEVELOPMENT_PHASES.items():
        if pillar in pillars:
            return phase
    return None

def get_complexity_for_task(task):
    """Get the complexity level for a specific task."""
    for complexity, tasks in TASK_COMPLEXITY.items():
        if task in tasks:
            return complexity
    return "UNKNOWN"

def get_available_tasks():
    """Get all available tasks with their metadata."""
    tasks = []
    for pillar, pillar_tasks in PILLARS.items():
        for task in pillar_tasks:
            phase = get_phase_for_pillar(pillar)
            complexity = get_complexity_for_task(task)
            tasks.append({
                "task": task,
                "pillar": pillar,
                "phase": phase,
                "complexity": complexity
            })
    return tasks

def get_tasks_by_phase(phase_name):
    """Get all tasks for a specific development phase."""
    if phase_name not in DEVELOPMENT_PHASES:
        return []
    
    phase_pillars = DEVELOPMENT_PHASES[phase_name]
    tasks = []
    for pillar in phase_pillars:
        if pillar in PILLARS:
            tasks.extend(PILLARS[pillar])
    return tasks

def get_tasks_by_complexity(complexity_level):
    """Get all tasks for a specific complexity level."""
    if complexity_level not in TASK_COMPLEXITY:
        return []
    return TASK_COMPLEXITY[complexity_level]

