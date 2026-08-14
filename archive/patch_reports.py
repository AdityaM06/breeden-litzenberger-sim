import re

with open('status.py', 'r') as f:
    content = f.read()

# Text Report Patch
text_old = r"    lines\.append\(\"\\n📊 1\. CAPITAL & RETURN PERFORMANCE\"\)"
text_new = """    lines.append("\\n⚙️ SIMULATION ASSUMPTIONS")
    lines.append("─" * 85)
    lines.append(f" - Allocation Model:    Concurrent Batch Allocation (grouped by entry date)")
    lines.append(f" - Position Sizing:     {POSITION_SIZE_PCT:.1%} of available deployable capital per trade")
    lines.append(f" - Max Positions:       {MAX_POSITIONS} concurrent open trades maximum")
    lines.append(f" - Cash Reserve:        {CASH_RESERVE_PCT:.1%} of equity kept uninvested")
    lines.append(f" - Short Borrow Cost:   {SHORT_BORROW_COST_ANNUAL:.1%} annualized fee for short positions")
    
    lines.append("\\n📊 1. CAPITAL & RETURN PERFORMANCE")"""
content = re.sub(text_old, text_new, content)

# Markdown Report Patch
md_old = r"    # Section 1: Capital & Return Overview\n    md\.append\(\"## 📊 1\. Capital & Return Overview\"\)"
md_new = """    # Simulation Assumptions
    md.append("## ⚙️ Simulation Assumptions")
    md.append("")
    md.append(f"- **Allocation Model**: Concurrent Batch Allocation (grouped by entry date)")
    md.append(f"- **Position Sizing**: `{POSITION_SIZE_PCT:.1%}` of available deployable capital per trade")
    md.append(f"- **Max Positions**: `{MAX_POSITIONS}` concurrent open trades maximum")
    md.append(f"- **Cash Reserve**: `{CASH_RESERVE_PCT:.1%}` of equity kept uninvested")
    md.append(f"- **Short Borrow Cost**: `{SHORT_BORROW_COST_ANNUAL:.1%}` annualized fee for short positions")
    md.append("")

    # Section 1: Capital & Return Overview
    md.append("## 📊 1. Capital & Return Overview")"""
content = re.sub(md_old, md_new, content)

# HTML Report Patch (Top navigation)
nav_old = r'<a href="#summary" class="nav-link">Executive Summary</a>'
nav_new = '<a href="#summary" class="nav-link">Executive Summary</a>\n                <a href="#assumptions" class="nav-link">Assumptions</a>'
content = content.replace(nav_old, nav_new)

# HTML Report Patch (Section)
html_old = r"        <!-- ============================================================= -->\n        <!-- SECTION 1: CAPITAL & RETURN KPIS -->"
html_new = """        <!-- ============================================================= -->
        <!-- SIMULATION ASSUMPTIONS -->
        <!-- ============================================================= -->
        <div class="section-header" id="assumptions">
            <div>
                <div class="section-title">⚙️ Simulation Assumptions</div>
                <div class="section-desc">Core constraints and parameters used to model realistic portfolio accounting.</div>
            </div>
        </div>
        <div class="callout" style="border-left-color: #a855f7;">
            <div class="callout-title" style="color: #a855f7;">Realistic Concurrent Simulation</div>
            <div class="callout-body">
                <ul style="margin-left: 1.5rem; margin-top: 0.5rem; color: var(--text-secondary);">
                    <li><strong>Allocation Model:</strong> Concurrent Batch Allocation (trades opening on same day share available capital).</li>
                    <li><strong>Position Sizing:</strong> Up to {POSITION_SIZE_PCT:.1%} of available capital per trade.</li>
                    <li><strong>Max Positions Cap:</strong> Maximum {MAX_POSITIONS} concurrent open trades.</li>
                    <li><strong>Cash Reserve:</strong> Minimum {CASH_RESERVE_PCT:.1%} of portfolio equity kept uninvested.</li>
                    <li><strong>Short Borrow Cost:</strong> {SHORT_BORROW_COST_ANNUAL:.1%} annualized fee subtracted daily from short positions.</li>
                </ul>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- SECTION 1: CAPITAL & RETURN KPIS -->"""
content = re.sub(html_old, html_new, content)

with open('status.py', 'w') as f:
    f.write(content)
print("Reports patched with Simulation Assumptions.")
