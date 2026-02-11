'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import ecosystemData from '@/market insights/ecosystem_visualization_data.json';

type ViewMode = 'interventions' | 'usecases';

export default function MarketRecallDepth() {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 1400, height: 700 });
  const [viewMode, setViewMode] = useState<ViewMode>('usecases');

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    // Prepare Market Insights data - process recall depth history
    const history = ecosystemData.recallDepthHistory;
    const providers = ['Google', 'Microsoft', 'OpenAI', 'Anthropic', 'Meta'];

    // Helper function to infer provider from model name
    const getProvider = (model: string): string => {
      const modelLower = model.toLowerCase();
      if (modelLower.includes('gpt') || modelLower.includes('o1') || modelLower.includes('o3')) return 'OpenAI';
      if (modelLower.includes('claude')) return 'Anthropic';
      if (modelLower.includes('gemini') || modelLower.includes('palm')) return 'Google';
      if (modelLower.includes('llama')) return 'Meta';
      if (modelLower.includes('azure') || modelLower.includes('copilot')) return 'Microsoft';
      return 'Unknown';
    };

    // Group by provider and get max depth per quarter
    const providerData = new Map<string, Map<string, number>>();
    history.forEach(item => {
      const provider = getProvider(item.model);
      if (provider === 'Unknown') return;

      if (!providerData.has(provider)) {
        providerData.set(provider, new Map());
      }
      const quarter = item.date;
      const current = providerData.get(provider)!.get(quarter) || 0;
      if (item.depth > current) {
        providerData.get(provider)!.set(quarter, item.depth);
      }
    });

    // Get all unique quarters
    const quarters = Array.from(new Set(history.map(h => h.date))).sort();

    // Create timeline with max depth per provider per quarter
    const timeline = quarters.map(quarter => {
      const dataPoint: any = { date: quarter };
      providers.forEach(provider => {
        const provMap = providerData.get(provider);
        let maxDepth = 0;
        quarters.filter(q => q <= quarter).forEach(q => {
          const depth = provMap?.get(q) || 0;
          if (depth > maxDepth) maxDepth = depth;
        });
        dataPoint[provider] = maxDepth || 0;
      });
      return dataPoint;
    });

    const realData = {
      providers,
      timeline
    };

    const { width, height } = dimensions;
    const margin = { top: 80, right: 20, bottom: 100, left: 100 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const colorScale = d3.scaleOrdinal()
      .domain(realData.providers)
      .range(['#4285f4', '#00a4ef', '#ff6b35', '#d4a574', '#00d4aa']);

    // Scales
    const xScale = d3.scalePoint()
      .domain(realData.timeline.map(d => d.date))
      .range([0, innerWidth])
      .padding(0.1);

    // Cap the scale at 1M to keep main data readable
    const yScale = d3.scaleLinear()
      .domain([0, 1000000])
      .range([innerHeight, 0]);

    // Find which providers have exceeded 1M
    const latestData = realData.timeline[realData.timeline.length - 1];
    const providersAbove1M = realData.providers.filter(p => latestData[p] > 1000000);

    // Add gridlines
    g.append('g')
      .attr('class', 'grid')
      .selectAll('line')
      .data(yScale.ticks(10))
      .join('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', d => yScale(d))
      .attr('y2', d => yScale(d))
      .attr('stroke', '#e2e8f0')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '2,2');

    // Critical 100K line
    const y100K = yScale(100000);
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', y100K)
      .attr('y2', y100K)
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5');

    g.append('text')
      .attr('x', innerWidth - 10)
      .attr('y', y100K - 8)
      .attr('text-anchor', 'end')
      .attr('fill', '#ef4444')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('100K Critical Threshold');

    // 1M boundary line (top of chart)
    const y1M = yScale(1000000);
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', y1M)
      .attr('y2', y1M)
      .attr('stroke', '#8b5cf6')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5');

    g.append('text')
      .attr('x', innerWidth - 10)
      .attr('y', y1M - 8)
      .attr('text-anchor', 'end')
      .attr('fill', '#8b5cf6')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('1M Context Window');

    // 10M line for Meta - positioned above the chart
    const y10M = -50; // Position above the chart in the expanded margin

    if (providersAbove1M.length > 0) {
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerWidth)
        .attr('y1', y10M)
        .attr('y2', y10M)
        .attr('stroke', '#00d4aa')
        .attr('stroke-width', 2.5)
        .attr('stroke-dasharray', '5,5');

      g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', y10M - 8)
        .attr('text-anchor', 'middle')
        .attr('fill', '#00d4aa')
        .style('font-size', '13px')
        .style('font-weight', '700')
        .text(`10M Context Window - ${providersAbove1M.join(', ')}`);
    }


    // X axis
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale));

    xAxis.selectAll('text')
      .style('font-size', '11px')
      .style('fill', '#475569')
      .attr('transform', 'rotate(-15)')
      .style('text-anchor', 'end');

    xAxis.select('.domain')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 2);

    // Y axis
    const yAxis = g.append('g')
      .call(d3.axisLeft(yScale).ticks(10).tickFormat(d => `${(d as number / 1000).toFixed(0)}K`));

    yAxis.selectAll('text')
      .style('font-size', '12px')
      .style('fill', '#475569');

    yAxis.select('.domain')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 2);

    // X axis label
    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 50)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', '600')
      .style('fill', '#334155')
      .text('Quarter');

    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -70)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', '600')
      .style('fill', '#334155')
      .text('Context Window (tokens)');

    // Line generator - clamp at 1M
    const line = d3.line<any>()
      .x(d => xScale(d.date) || 0)
      .y((d, _i, provider: any) => yScale(Math.min(d[provider], 1000000)))
      .curve(d3.curveMonotoneX);

    // Draw lines and bubbles for each provider
    realData.providers.forEach((provider, seriesIndex) => {
      const safeId = `series-${seriesIndex}`;
      const isAbove1M = providersAbove1M.includes(provider);

      // Helper function to get y position - if above 1M, use 10M line position
      const getYPosition = (value: number) => {
        if (isAbove1M && value > 1000000) {
          return y10M; // Plot at 10M line
        }
        return yScale(Math.min(value, 1000000)); // Normal scale or clamped at 1M
      };

      // Line path
      g.append('path')
        .datum(realData.timeline)
        .attr('fill', 'none')
        .attr('stroke', colorScale(provider) as string)
        .attr('stroke-width', 2.5)
        .attr('d', line.y(d => getYPosition(d[provider as keyof typeof d] as number)))
        .style('opacity', 0.8);

      // Bubbles at each data point
      g.selectAll(`.bubble-${safeId}`)
        .data(realData.timeline)
        .join('circle')
        .attr('class', `bubble-${safeId}`)
        .attr('cx', d => xScale(d.date) || 0)
        .attr('cy', d => getYPosition(d[provider as keyof typeof d] as number))
        .attr('r', 5)
        .attr('fill', colorScale(provider) as string)
        .attr('stroke', 'white')
        .attr('stroke-width', 2)
        .style('opacity', 0.9)
        .style('cursor', 'pointer')
        .on('mouseover', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', 7)
            .style('opacity', 1);
        })
        .on('mouseout', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', 5)
            .style('opacity', 0.9);
        });
    });


  }, [viewMode, dimensions]);

  return (
    <div className="flex flex-col w-full h-full">
      {/* Legend Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <div className="text-sm font-medium text-slate-700">
          Track context window growth - when models exceed 100K tokens, they don't need external memory
        </div>

        {/* Line Color Legend */}
        <div className="flex gap-3 text-xs">
          {['Google', 'Microsoft', 'OpenAI', 'Anthropic', 'Meta'].map((provider, i) => {
            const colors = ['#4285f4', '#00a4ef', '#ff6b35', '#d4a574', '#00d4aa'];
            return (
              <div key={provider} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: colors[i] }}
                ></div>
                <span className="text-slate-700">{provider}</span>
              </div>
            );
          })}
        </div>
      </div>


      {/* Chart */}
      <div ref={containerRef} className="flex-1 w-full min-h-[700px]">
        <svg ref={svgRef}></svg>
      </div>
    </div>
  );
}
