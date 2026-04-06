'use client';

import { useState } from 'react';
import MarketMemoryGrowth from '@/components/MarketMemoryGrowth';
import MarketAgentVelocity from '@/components/MarketAgentVelocity';
import MarketOpennessMatrix from '@/components/MarketOpennessMatrix';
import MarketRecallDepth from '@/components/MarketRecallDepth';

type ViewType = 'intro' | 'market-memory-growth' | 'market-agent-velocity' | 'market-openness-matrix' | 'market-recall-depth';

export default function Home() {
  const [activeView, setActiveView] = useState<ViewType>('intro');
  const [marketOpennessViewMode, setMarketOpennessViewMode] = useState<'matrix' | 'breakdown' | 'timeline'>('matrix');

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-50">
      {/* Header */}
      <header className="flex-none bg-gradient-to-r from-teal-700 to-emerald-700 text-white shadow-lg z-50">
        <div className="px-8 py-6 flex items-center justify-between">
          <h1 className="text-3xl font-semibold tracking-tight">AI Market Insights Dashboard</h1>

          {/* Dropdown */}
          <div className="relative inline-block">
            <select
              value={activeView}
              onChange={(e) => {
                setActiveView(e.target.value as ViewType);
              }}
              className="appearance-none bg-white text-slate-900 px-6 py-2 pr-10 rounded-lg font-medium shadow-md border-2 border-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-400 cursor-pointer"
            >
              <option value="intro">Introduction</option>
              <optgroup label="Market Insights">
                <option value="market-memory-growth">Memory-Related Product Growth</option>
                <option value="market-agent-velocity">Agent Capabilities Growth</option>
                <option value="market-openness-matrix">Memory Openness vs Lock-In</option>
                <option value="market-recall-depth">AI Context Window Growth</option>
              </optgroup>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-slate-700">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
              </svg>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      {activeView === 'intro' ? (
        /* Introduction Page */
        <div className="flex-1 overflow-y-auto bg-white">
          <div className="max-w-5xl mx-auto px-12 py-16">
            {/* Header Section */}
            <div className="mb-16">
              <h2 className="text-4xl font-light text-slate-900 mb-4 tracking-tight">
                AI Market Insights Dashboard
              </h2>
              <p className="text-lg text-slate-600 leading-relaxed mb-4">
                A strategic framework for analyzing the AI infrastructure landscape through data-driven visualizations of hyperscaler momentum, agent capabilities, openness strategies, and technical evolution.
              </p>

              {/* Data Update Note */}
              <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded-r-lg">
                <p className="text-sm text-amber-900 leading-relaxed">
                  <strong>Note:</strong> All data shown is currently manually updated. We're piloting an automated insight update system for the next release cycle.
                </p>
              </div>
            </div>

            {/* What We Track */}
            <div className="mb-12">
              <h3 className="text-2xl font-semibold text-slate-900 mb-4">What We Track</h3>
              <p className="text-slate-600 mb-8 leading-relaxed">
                This dashboard provides comprehensive analysis across four key dimensions of the AI market, with a focus on memory infrastructure and strategic positioning of major hyperscalers.
              </p>

              <div className="grid grid-cols-1 gap-4 mb-12">
                <div className="border-l-4 border-teal-500 pl-6 py-3">
                  <h4 className="font-semibold text-slate-900">Memory-Related Product Growth</h4>
                  <p className="text-sm text-slate-600">Track the evolution of memory-related features across major AI providers, comparing memory updates to total product releases over time.</p>
                </div>
                <div className="border-l-4 border-emerald-500 pl-6 py-3">
                  <h4 className="font-semibold text-slate-900">Agent Capabilities Growth</h4>
                  <p className="text-sm text-slate-600">Monitor the velocity of agent feature releases across Google, Microsoft, OpenAI, Anthropic, and Meta.</p>
                </div>
                <div className="border-l-4 border-cyan-500 pl-6 py-3">
                  <h4 className="font-semibold text-slate-900">Memory Openness vs Lock-In</h4>
                  <p className="text-sm text-slate-600">Analyze strategic positioning through three lenses: Strategy Matrix, Factor Breakdown, and Momentum Timeline.</p>
                </div>
                <div className="border-l-4 border-sky-500 pl-6 py-3">
                  <h4 className="font-semibold text-slate-900">AI Context Window Growth</h4>
                  <p className="text-sm text-slate-600">Explore the evolution of model context windows, including breakthrough milestones like Meta's 10M token achievement.</p>
                </div>
              </div>
            </div>

            {/* Key Providers */}
            <div className="mb-12">
              <h3 className="text-2xl font-semibold text-slate-900 mb-4">Providers Tracked</h3>
              <p className="text-slate-600 mb-6 leading-relaxed">
                We monitor the five major AI hyperscalers shaping the industry:
              </p>

              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="bg-gradient-to-br from-blue-50 to-white border border-blue-200 p-4 rounded-lg text-center">
                  <div className="w-3 h-3 rounded-full bg-[#4285f4] mx-auto mb-2"></div>
                  <h4 className="font-semibold text-slate-900 text-sm">Google</h4>
                </div>
                <div className="bg-gradient-to-br from-sky-50 to-white border border-sky-200 p-4 rounded-lg text-center">
                  <div className="w-3 h-3 rounded-full bg-[#00a4ef] mx-auto mb-2"></div>
                  <h4 className="font-semibold text-slate-900 text-sm">Microsoft</h4>
                </div>
                <div className="bg-gradient-to-br from-orange-50 to-white border border-orange-200 p-4 rounded-lg text-center">
                  <div className="w-3 h-3 rounded-full bg-[#ff6b35] mx-auto mb-2"></div>
                  <h4 className="font-semibold text-slate-900 text-sm">OpenAI</h4>
                </div>
                <div className="bg-gradient-to-br from-amber-50 to-white border border-amber-200 p-4 rounded-lg text-center">
                  <div className="w-3 h-3 rounded-full bg-[#d4a574] mx-auto mb-2"></div>
                  <h4 className="font-semibold text-slate-900 text-sm">Anthropic</h4>
                </div>
                <div className="bg-gradient-to-br from-teal-50 to-white border border-teal-200 p-4 rounded-lg text-center">
                  <div className="w-3 h-3 rounded-full bg-[#00d4aa] mx-auto mb-2"></div>
                  <h4 className="font-semibold text-slate-900 text-sm">Meta</h4>
                </div>
              </div>
            </div>

            {/* Get Started */}
            <div className="bg-gradient-to-r from-teal-600 to-emerald-600 text-white p-8 rounded-lg shadow-lg">
              <h3 className="text-2xl font-semibold mb-3">Get Started</h3>
              <p className="mb-6 text-teal-50">
                Use the dropdown menu in the header to explore each visualization. Each view includes detailed insights and methodology notes in the sidebar.
              </p>
              <button
                onClick={() => setActiveView('market-memory-growth')}
                className="bg-white text-teal-700 hover:bg-teal-50 font-medium px-6 py-3 rounded-lg transition-colors shadow-md"
              >
                Start Exploring →
              </button>
            </div>
          </div>
        </div>
      ) : (
        /* Visualization Views */
        <div className="flex flex-1 overflow-hidden">
          {/* Main Chart Area */}
          <div className="flex-1 flex flex-col overflow-hidden min-w-0">
            <div className="flex-1 flex items-center justify-center p-6 bg-white">
              {activeView === 'market-memory-growth' && <MarketMemoryGrowth />}
              {activeView === 'market-agent-velocity' && <MarketAgentVelocity />}
              {activeView === 'market-openness-matrix' && (
                <MarketOpennessMatrix
                  viewMode={marketOpennessViewMode}
                  onViewModeChange={setMarketOpennessViewMode}
                />
              )}
              {activeView === 'market-recall-depth' && <MarketRecallDepth />}
            </div>
          </div>

          {/* Right Sidebar - Insights */}
          <aside className="w-96 min-w-96 max-w-96 flex-shrink-0 bg-slate-50 border-l border-slate-200 overflow-y-auto overflow-x-hidden p-6">
            {activeView === 'market-memory-growth' ? (
              <div>
                <details open className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Q4 2025 → Feb 2026 Insights
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words">Memory innovation accelerated in late 2025 with 10 new products launched between May-December, bringing total tracked products to 51. Q4 2025 saw memory features mature from experimental to production-grade, with enterprise governance tools (webhooks, policy APIs, compliance scanners) dominating releases alongside consumer memory packs and scenario-scoped memories.</p>
                    <p className="break-words"><strong>Key Insights:</strong></p>
                    <p className="break-words">• Memory share peaked at 25% of total hyperscaler updates in 2025, up from 18% in 2024</p>
                    <p className="break-words">• Governance wave in Q4 2025: Memory webhooks (Anthropic), Memory Policy API (OpenAI), Risk Dashboard (Microsoft)</p>
                    <p className="break-words">• Consumer packaging: OpenAI Advisor Memory Packs (Oct 2025) democratize pre-configured memory templates</p>
                    <p className="break-words">• Education-specific memory: Google Classroom Long-Term Memory (Feb 2025), Teams EDU Memory Assistant show vertical specialization</p>
                    <p className="break-words">• 2026 slowdown signal: Only 4 events in January suggests post-holiday pause or market maturation</p>
                  </div>
                </details>

                <details className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Purpose & Methodology
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words"><strong>Purpose:</strong> Track the evolution and adoption of persistent memory features across hyperscalers, measuring both product launches and memory's share of total platform updates to understand strategic prioritization.</p>
                    <p className="break-words"><strong>Methodology:</strong> Counted memory-specific product releases from database events (categories: user memory, developer memory, governance tools). Memory share calculated as (memory updates / total updates) by year. Tracked 2023-2026 to capture acceleration from experimental (2023-2024) to production-grade (2025-2026).</p>
                  </div>
                </details>
              </div>
            ) : activeView === 'market-agent-velocity' ? (
              <div>
                <details open className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Q4 2025 → Feb 2026 Insights
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words">Agentic capabilities exploded in late 2025 with 18 new features launched in Q4 alone, driven by OpenAI Frontier (Feb 2026), Anthropic Agent Teams (Dec 2025), and Microsoft Agent 365 (Nov 2025). The market shifted from "agents as demos" to "agents as products" with production-ready orchestration, team collaboration, and enterprise deployment platforms.</p>
                    <p className="break-words"><strong>Key Insights:</strong></p>
                    <p className="break-words">• 60 total agentic features tracked across hyperscalers (2023-2026), with 70% launched in 2025-2026</p>
                    <p className="break-words">• OpenAI late surge: 3 major releases in 6 months (Codex 5.3, AgentKit, Frontier) after slow 2024</p>
                    <p className="break-words">• Anthropic leads 2025: 11 features including Agent Teams, Cowork, Claude Code, MCP donation to AAIF</p>
                    <p className="break-words">• Microsoft enterprise focus: Agent 365, Foundry IQ, Fabric IQ target enterprise workflows</p>
                    <p className="break-words">• Definition matters: Excludes model releases, counts only platforms, tools, features, frameworks with multi-step agentic capabilities</p>
                  </div>
                </details>

                <details className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Purpose & Methodology
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words"><strong>Purpose:</strong> Measure competitive intensity in agentic AI by tracking platforms, tools, features, and frameworks that enable multi-step reasoning, planning, tool orchestration, and goal persistence—distinct from base model improvements.</p>
                    <p className="break-words"><strong>Methodology:</strong> Counted agentic feature releases excluding model launches, context window increases, and non-agent memory features. Inclusion criteria: agent builders, deployment platforms, code execution, computer control, custom agents, workflows, orchestration SDKs, multi-agent systems. Data from systematic database research (2023-2026).</p>
                  </div>
                </details>
              </div>
            ) : activeView === 'market-recall-depth' ? (
              <div>
                <details open className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Q4 2025 → Feb 2026 Insights
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words">Context windows stabilized in late 2025 after Llama 4 Scout's 10M token breakthrough (April 2025), with subsequent releases focusing on model quality over raw capacity. GPT-5 series maintained 400K tokens across three variants (Aug-Dec 2025), while Claude 4.6 Opus (Feb 2026) and Gemini 3 series held at 200K and 1M respectively, suggesting market consensus on "good enough" thresholds.</p>
                    <p className="break-words"><strong>Key Insights:</strong></p>
                    <p className="break-words">• Llama 4 Scout's 10M tokens (April 2025) represents 50x jump from GPT-4's 200K, but open source only</p>
                    <p className="break-words">• Commercial models plateau: GPT-5.x at 400K, Claude at 200K, Gemini at 1M through Feb 2026</p>
                    <p className="break-words">• No new breakthroughs in 10 months: Last major increase was Llama 4 Scout, suggesting focus shifted to quality/reasoning</p>
                    <p className="break-words">• Reasoning vs recall trade-off: Extended thinking modes (Claude, Gemini) prioritize depth over breadth</p>
                    <p className="break-words">• Cost constraints: Larger context = higher inference costs may be limiting commercial expansion</p>
                  </div>
                </details>

                <details className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Purpose & Methodology
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words"><strong>Purpose:</strong> Track the evolution of input token limits (context windows) across frontier models to understand each hyperscaler's approach to "recall depth"—how much information models can consider in a single request.</p>
                    <p className="break-words"><strong>Methodology:</strong> Recorded maximum input context window sizes for major model releases from database and web research. Focused on production models (excluded research previews unless widely available). Tracked 2023-2026 to capture progression from 8K (GPT-4 launch) to 10M (Llama 4 Scout). Context window represents input tokens only, not output limits.</p>
                  </div>
                </details>
              </div>
            ) : activeView === 'market-openness-matrix' && marketOpennessViewMode === 'matrix' ? (
              <div>
                <details open className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Q4 2025 → Feb 2026 Insights
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words">Strategic divergence on memory reached a tipping point in late 2025 as Anthropic donated MCP to the Linux Foundation (Dec 2025) while OpenAI's CSO explicitly confirmed ecosystem-only portability (Feb 2025). The competitive landscape now shows clear archetypal splits between moat builders (OpenAI, Google) and bridge builders (Anthropic, Microsoft).</p>
                    <p className="break-words"><strong>Key Insights:</strong></p>
                    <p className="break-words">• Anthropic leads on openness (8.5/10) after donating MCP to AAIF, establishing industry standard</p>
                    <p className="break-words">• OpenAI highest lock-in (8.0/10) with CSO's explicit "within our ecosystem, not across platforms" stance</p>
                    <p className="break-words">• Google's conflicted position (7.0 lock-in, 4.0 openness) reflects "two minds" strategy—ecosystem power vs late MCP adoption</p>
                    <p className="break-words">• Microsoft investing in portability (7.5 openness) with CTO dedicating 2 FTEs to portable semantic memory</p>
                    <p className="break-words">• Meta opts out (2.0 lock-in) of enterprise memory market entirely, betting on open source developer-managed approach</p>
                  </div>
                </details>

                <details className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Purpose & Methodology
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words"><strong>Purpose:</strong> Visualize competitive memory strategies as Lock-In Index (barriers to leaving) vs Openness Score (portability + user agency), revealing which hyperscalers build retention moats vs integration bridges.</p>
                    <p className="break-words"><strong>Methodology:</strong> Scored each provider on Lock-In Index (switching costs 40%, data portability 30%, user control 30%) and Openness Score (data portability 40%, interoperability standards 35%, user control 25%) using database events, web research, and director's insider context. Scores range 0-10; lock-in lower is better, openness higher is better.</p>
                  </div>
                </details>
              </div>
            ) : activeView === 'market-openness-matrix' && marketOpennessViewMode === 'breakdown' ? (
              <div>
                <details open className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Q4 2025 → Feb 2026 Insights
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words">Factor-level analysis reveals strategic trade-offs: OpenAI maximizes switching costs (4.0/4.0) while minimizing data portability (0.8/4.0). Anthropic inverts this with maximum interoperability (3.5/3.5) through MCP donation. Microsoft balances across all factors, while Google's high switching costs (3.0/4.0) from ecosystem integration contradict their late openness gestures.</p>
                    <p className="break-words"><strong>Key Insights:</strong></p>
                    <p className="break-words">• Switching Costs drive lock-in: OpenAI (4.0) and Google (3.0) create maximum migration barriers through ecosystem integration</p>
                    <p className="break-words">• Interoperability defines openness leaders: Anthropic (3.5/3.5) created MCP standard; Microsoft (2.7/3.5) joined AAIF</p>
                    <p className="break-words">• Data Portability gap: Anthropic (3.2/4.0) vs OpenAI (0.8/4.0) shows 4x difference in export capabilities</p>
                    <p className="break-words">• User Control varies: Microsoft leads on enterprise controls (2.0/2.5); OpenAI lags on consumer transparency (1.2/2.5)</p>
                    <p className="break-words">• Meta's anomaly: Low scores across lock-in factors (0.5 switching costs, 0.5 portability) because no persistent memory offering</p>
                  </div>
                </details>

                <details className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Purpose & Methodology
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words"><strong>Purpose:</strong> Decompose composite Lock-In and Openness scores into component factors (switching costs, data portability, interoperability, user control) to reveal where strategic differences occur and why providers score as they do.</p>
                    <p className="break-words"><strong>Methodology:</strong> Each provider scored 0-10 on Lock-In factors (switching costs, data portability, user control) and Openness factors (data portability, interoperability standards, user control). Factor scores sum to composite totals. Weights reflect relative importance: switching costs highest for lock-in, data portability + interoperability highest for openness. Based on memory feature analysis, API capabilities, protocol adoptions, and governance tools.</p>
                  </div>
                </details>
              </div>
            ) : activeView === 'market-openness-matrix' && marketOpennessViewMode === 'timeline' ? (
              <div>
                <details open className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Q4 2025 → Feb 2026 Insights
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words">The openness gap widened dramatically in Q4 2025 as competitive dynamics accelerated. Anthropic's MCP donation to AAIF (Dec 2025) cemented its openness leadership while OpenAI's lock-in strategy intensified. Microsoft's openness trajectory continued upward following Memory Export API launch (May 2024), while Google remained stagnant despite late MCP adoption (April 2025).</p>
                    <p className="break-words"><strong>Key Insights:</strong></p>
                    <p className="break-words">• Anthropic's openness surge from 4.0 (2023) → 8.5 (2026) driven by MCP creation and persistent memory with granular controls</p>
                    <p className="break-words">• OpenAI's lock-in intensification from 6.0 (2023) → 8.0 (2026) as memory became explicit retention moat</p>
                    <p className="break-words">• Microsoft's steady openness climb from 4.0 (2023) → 7.5 (2026) through Memory Export API and portable memory investment</p>
                    <p className="break-words">• Google's plateau at 7.0 lock-in with minimal openness movement (3.0 → 4.0) reflects strategic paralysis</p>
                    <p className="break-words">• MCP adoption cascade (Nov 2024 - April 2025) forced competitive responses but didn't shift OpenAI's core strategy</p>
                  </div>
                </details>

                <details className="mb-6 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <summary className="px-4 py-3 cursor-pointer font-semibold text-slate-900 hover:bg-slate-50 transition-colors rounded-lg">
                    Purpose & Methodology
                  </summary>
                  <div className="px-4 py-4 space-y-4 text-sm text-slate-700 leading-relaxed">
                    <p className="break-words"><strong>Purpose:</strong> Track how Lock-In and Openness scores evolved quarterly from 2023-2026, revealing momentum shifts and competitive responses to major events (MCP launch, CSO quotes, API releases).</p>
                    <p className="break-words"><strong>Methodology:</strong> Quarterly scores calculated by analyzing memory feature releases, protocol adoptions, and policy announcements in database. Key events annotated with provider, date, and impact type. Timeline shows whether "bridge strategy" (openness) is gaining ground vs "moat strategy" (lock-in) over time.</p>
                  </div>
                </details>
              </div>
            ) : null}
          </aside>
        </div>
      )}
    </div>
  );
}
