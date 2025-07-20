# Telegram Bot Research Assistant - Migration Plan

## Flask to Gradio Migration

### Current State
- Flask-based web interface with templates and static folders
- Telegram bot integration with MCP research server
- Function calling infrastructure for automatic tool invocation
- Clean UX with removed MCP Tools menus

### Migration Considerations

#### Templates and Static Folders
- **NOT NEEDED** when migrating from Flask to Gradio
- Gradio automatically generates web interface and handles static assets
- Can safely delete `templates/` and `static/` folders
- All UI defined in Python code using Gradio components

#### What Changes
- Replace Flask routes with Gradio interface components
- Convert HTML templates to Gradio blocks/interfaces
- Remove static file serving logic
- Maintain backend MCP integration and Telegram bot functionality

#### What Stays the Same
- MCP client integration
- Telegram bot handlers
- Function calling infrastructure
- Research tool capabilities

### Implementation Steps
1. Install Gradio dependency
2. Create new Gradio interface for research functionality
3. Migrate Flask endpoints to Gradio event handlers
4. Remove Flask-specific code and dependencies
5. Clean up templates and static folders
6. Test web interface functionality
7. Update deployment configuration

### Benefits
- Simpler UI development with Python-only interface
- Automatic responsive design
- Built-in component library
- Easier deployment and maintenance
- No need for HTML/CSS/JS knowledge

### Notes
- Keep custom assets only if specifically needed for branding
- Gradio handles all web serving internally
- Focus on Python-based UI components instead of templates
