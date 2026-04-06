'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import ecosystemData from '@/market-insights/ecosystem_visualization_data.json';

type ViewMode = 'provider' | 'aggregate';

export default function MarketMemoryGrowth() {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 1400, height: 700 });
  const [viewMode, setViewMode] = useState<ViewMode>('provider');

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

    const { width, height } = dimensions;
    const margin = viewMode === 'aggregate'
      ? { top: 40, right: 100, bottom: 100, left: 80 }
      : { top: 40, right: 20, bottom: 100, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    if (viewMode === 'provider') {
      // View 1: Memory vs Total Updates by Provider
      const allUpdates = ecosystemData.allUpdatesByYear;
      const years = Object.keys(allUpdates).filter(y => parseInt(y) <= 2025);
      const providers = ['Google', 'Microsoft', 'OpenAI', 'Anthropic', 'Meta'];

      // Prepare data for grouped bars
      const barData = years.map(year => {
        const yearData = allUpdates[year as keyof typeof allUpdates];
        return {
          year,
          providers: providers.map(provider => ({
            provider,
            total: yearData[provider as keyof typeof yearData]?.total || 0,
            memory: yearData[provider as keyof typeof yearData]?.memory || 0
          }))
        };
      });

      const colorScale = d3.scaleOrdinal()
        .domain(providers)
        .range(['#4285f4', '#00a4ef', '#ff6b35', '#d4a574', '#00d4aa']);

      const xScale = d3.scaleBand()
        .domain(years)
        .range([0, innerWidth])
        .padding(0.2);

      const maxValue = d3.max(barData, d =>
        d3.max(d.providers, p => p.total)
      ) || 0;

      const yScale = d3.scaleLinear()
        .domain([0, maxValue * 1.2])
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
        .text('Number of Product Updates');

      // Grouped bars - total and memory for each provider
      const barWidth = xScale.bandwidth() / providers.length;
      const memoryBarWidth = barWidth * 0.4;

      barData.forEach(yearData => {
        yearData.providers.forEach((providerData, pIdx) => {
          const x = (xScale(yearData.year) || 0) + pIdx * barWidth;

          // Total bar (lighter)
          if (providerData.total > 0) {
            g.append('rect')
              .attr('x', x)
              .attr('y', yScale(providerData.total))
              .attr('width', barWidth - 2)
              .attr('height', innerHeight - yScale(providerData.total))
              .attr('fill', colorScale(providerData.provider) as string)
              .attr('rx', 4)
              .style('opacity', 0.3)
              .style('cursor', 'pointer')
              .on('mouseover', function() {
                d3.select(this).style('opacity', 0.5);
              })
              .on('mouseout', function() {
                d3.select(this).style('opacity', 0.3);
              });
          }

          // Memory bar (darker, on top)
          if (providerData.memory > 0) {
            g.append('rect')
              .attr('x', x + (barWidth - memoryBarWidth) / 2)
              .attr('y', yScale(providerData.memory))
              .attr('width', memoryBarWidth)
              .attr('height', innerHeight - yScale(providerData.memory))
              .attr('fill', colorScale(providerData.provider) as string)
              .attr('rx', 3)
              .style('opacity', 0.9)
              .style('cursor', 'pointer')
              .on('mouseover', function() {
                d3.select(this).style('opacity', 1);
              })
              .on('mouseout', function() {
                d3.select(this).style('opacity', 0.9);
              });

            // Memory value label
            g.append('text')
              .attr('x', x + barWidth / 2)
              .attr('y', yScale(providerData.memory) - 5)
              .attr('text-anchor', 'middle')
              .attr('fill', '#1e293b')
              .style('font-size', '10px')
              .style('font-weight', '700')
              .text(providerData.memory);
          }
        });
      });

    } else {
      // View 2: Memory Share by Year (Aggregate)
      const memoryShare = ecosystemData.memoryShareByYear;
      const years = Object.keys(memoryShare).filter(y => parseInt(y) <= 2025);

      const data = years.map(year => {
        const yearData = memoryShare[year as keyof typeof memoryShare];
        return {
          year,
          totalUpdates: yearData.totalUpdates,
          memoryUpdates: yearData.memoryUpdates,
          memoryShare: yearData.memoryShare
        };
      });

      const xScale = d3.scaleBand()
        .domain(years)
        .range([0, innerWidth])
        .padding(0.3);

      const yScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.totalUpdates) || 0])
        .range([innerHeight, 0]);

      const yScalePercent = d3.scaleLinear()
        .domain([0, 100])
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

      // Y axis (left - counts)
      const yAxis = g.append('g')
        .call(d3.axisLeft(yScale).ticks(10));

      yAxis.selectAll('text')
        .style('font-size', '12px')
        .style('fill', '#475569');

      yAxis.select('.domain')
        .attr('stroke', '#cbd5e1')
        .attr('stroke-width', 2);

      // Y axis (right - percentage)
      const yAxisRight = g.append('g')
        .attr('transform', `translate(${innerWidth},0)`)
        .call(d3.axisRight(yScalePercent).ticks(10).tickFormat(d => `${d}%`));

      yAxisRight.selectAll('text')
        .style('font-size', '12px')
        .style('fill', '#ef4444');

      yAxisRight.select('.domain')
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

      // Y axis label (left)
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -55)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .style('fill', '#334155')
        .text('Number of Updates');

      // Y axis label (right)
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', innerWidth + 75)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .style('fill', '#ef4444')
        .text('Memory Share (%)');

      // Stacked bars
      const barWidth = xScale.bandwidth();

      data.forEach(d => {
        const x = xScale(d.year) || 0;

        // Total updates bar
        g.append('rect')
          .attr('x', x)
          .attr('y', yScale(d.totalUpdates))
          .attr('width', barWidth)
          .attr('height', innerHeight - yScale(d.totalUpdates))
          .attr('fill', '#cbd5e1')
          .attr('rx', 4)
          .style('opacity', 0.6);

        // Memory updates bar (on top)
        g.append('rect')
          .attr('x', x)
          .attr('y', yScale(d.memoryUpdates))
          .attr('width', barWidth)
          .attr('height', innerHeight - yScale(d.memoryUpdates))
          .attr('fill', '#3b82f6')
          .attr('rx', 4)
          .style('opacity', 0.8);

        // Memory share line point
        g.append('circle')
          .attr('cx', x + barWidth / 2)
          .attr('cy', yScalePercent(d.memoryShare))
          .attr('r', 6)
          .attr('fill', '#ef4444')
          .attr('stroke', 'white')
          .attr('stroke-width', 2);

        // Labels
        g.append('text')
          .attr('x', x + barWidth / 2)
          .attr('y', yScale(d.memoryUpdates) - 5)
          .attr('text-anchor', 'middle')
          .attr('fill', '#1e293b')
          .style('font-size', '11px')
          .style('font-weight', '700')
          .text(`${d.memoryUpdates}/${d.totalUpdates}`);

        g.append('text')
          .attr('x', x + barWidth / 2)
          .attr('y', yScalePercent(d.memoryShare) - 12)
          .attr('text-anchor', 'middle')
          .attr('fill', '#ef4444')
          .style('font-size', '11px')
          .style('font-weight', '700')
          .text(`${d.memoryShare.toFixed(1)}%`);
      });

      // Memory share line
      const line = d3.line<any>()
        .x(d => (xScale(d.year) || 0) + xScale.bandwidth() / 2)
        .y(d => yScalePercent(d.memoryShare))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', '#ef4444')
        .attr('stroke-width', 2.5)
        .attr('d', line)
        .style('opacity', 0.8);
    }

  }, [viewMode, dimensions]);

  return (
    <div className="flex flex-col w-full h-full">
      {/* Legend Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <div className="flex items-center gap-4">
          <div className="text-sm font-medium text-slate-700">
            {viewMode === 'provider'
              ? 'Memory vs Total Updates by Provider'
              : 'Memory Share of Total Updates (Hyperscaler-Agnostic)'}
          </div>

          {/* View Toggle */}
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode('provider')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                viewMode === 'provider'
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              By Provider
            </button>
            <button
              onClick={() => setViewMode('aggregate')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                viewMode === 'aggregate'
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Aggregate
            </button>
          </div>
        </div>

        {/* Line Color Legend */}
        {viewMode === 'provider' && (
          <div className="flex gap-3 text-xs">
            {['Google', 'Microsoft', 'OpenAI', 'Anthropic', 'Meta'].map((provider, i) => {
              const colors = ['#4285f4', '#00a4ef', '#ff6b35', '#d4a574', '#00d4aa'];
              return (
                <div key={provider} className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded flex-shrink-0"
                    style={{ backgroundColor: colors[i] }}
                  ></div>
                  <span className="text-slate-700">{provider}</span>
                </div>
              );
            })}
          </div>
        )}
        {viewMode === 'aggregate' && (
          <div className="flex gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded flex-shrink-0 bg-slate-400"></div>
              <span className="text-slate-700">Total Updates</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded flex-shrink-0 bg-blue-500"></div>
              <span className="text-slate-700">Memory Updates</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full flex-shrink-0 bg-red-500"></div>
              <span className="text-slate-700">Memory Share %</span>
            </div>
          </div>
        )}
      </div>


      {/* Chart */}
      <div ref={containerRef} className="flex-1 w-full min-h-[700px]">
        <svg ref={svgRef}></svg>
      </div>
    </div>
  );
}
