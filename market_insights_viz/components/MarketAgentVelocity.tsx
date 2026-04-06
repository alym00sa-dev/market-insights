'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import ecosystemData from '@/market-insights/ecosystem_visualization_data.json';

type ViewMode = 'interventions' | 'usecases';

export default function MarketAgentVelocity() {
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

    // Process agentFeatures data from JSON
    const agentFeatures = ecosystemData.agentFeatures;
    const years = ['2023', '2024', '2025', '2026'];
    const providers = ['Google', 'Microsoft', 'OpenAI', 'Anthropic', 'Meta'];

    // Create timeline by year (non-cumulative)
    const timeline = years.map(year => {
      const dataPoint: any = { date: year };
      providers.forEach(provider => {
        const providerData = agentFeatures.find(af => af.provider === provider);
        if (providerData) {
          // Use the year's value directly (non-cumulative)
          dataPoint[provider] = (providerData as any)[year] || 0;
        } else {
          dataPoint[provider] = 0;
        }
      });
      return dataPoint;
    });

    const realData = {
      providers,
      timeline
    };

    const { width, height } = dimensions;
    const margin = { top: 40, right: 20, bottom: 100, left: 80 };
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

    const maxValue = d3.max(realData.timeline, d =>
      Math.max(...realData.providers.map(p => d[p as keyof typeof d] as number))
    ) || 0;

    const yScale = d3.scaleLinear()
      .domain([0, maxValue * 1.1])
      .range([innerHeight, 0]);

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

    // X axis
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale));

    xAxis.selectAll('text')
      .style('font-size', '11px')
      .style('fill', '#475569');

    xAxis.select('.domain')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 2);

    // Y axis
    const yAxis = g.append('g')
      .call(d3.axisLeft(yScale).ticks(10));

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
      .text('Year');

    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -55)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', '600')
      .style('fill', '#334155')
      .text('Agent Features Released');

    // Line generator
    const line = d3.line<any>()
      .x(d => xScale(d.date) || 0)
      .y((d, _i, provider: any) => yScale(d[provider]))
      .curve(d3.curveMonotoneX);

    // Draw lines and bubbles for each provider
    realData.providers.forEach((provider, seriesIndex) => {
      const safeId = `series-${seriesIndex}`;

      // Line path
      g.append('path')
        .datum(realData.timeline)
        .attr('fill', 'none')
        .attr('stroke', colorScale(provider) as string)
        .attr('stroke-width', 2.5)
        .attr('d', line.y(d => yScale(d[provider as keyof typeof d] as number)))
        .style('opacity', 0.8);

      // Bubbles at each data point
      g.selectAll(`.bubble-${safeId}`)
        .data(realData.timeline)
        .join('circle')
        .attr('class', `bubble-${safeId}`)
        .attr('cx', d => xScale(d.date) || 0)
        .attr('cy', d => yScale(d[provider as keyof typeof d] as number))
        .attr('r', 5)
        .attr('fill', colorScale(provider) as string)
        .attr('stroke', 'white')
        .attr('stroke-width', 2)
        .style('opacity', 0.9)
        .style('cursor', 'pointer')
        .on('mouseover', function(_event, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', 7)
            .style('opacity', 1);
        })
        .on('mouseout', function(_event, d) {
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
          Track agent feature release velocity across major AI providers
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
