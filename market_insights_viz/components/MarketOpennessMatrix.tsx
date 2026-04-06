'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import ecosystemData from '@/market-insights/ecosystem_visualization_data.json';

type ViewMode = 'matrix' | 'breakdown' | 'timeline';

interface MarketOpennessMatrixProps {
  viewMode?: ViewMode;
  onViewModeChange?: (mode: ViewMode) => void;
}

export default function MarketOpennessMatrix({ viewMode: externalViewMode = 'matrix', onViewModeChange }: MarketOpennessMatrixProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 1400, height: 700 });
  const viewMode = externalViewMode;

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

    if (viewMode === 'matrix') {
      // View 1: Strategy Matrix - Scatter plot
      const providers = ecosystemData.lockInOpenness.strategyMatrix;

      // Scales
      const xScale = d3.scaleLinear()
        .domain([0, 10])
        .range([0, innerWidth]);

      const yScale = d3.scaleLinear()
        .domain([0, 10])
        .range([innerHeight, 0]);

      // Quadrant backgrounds
      const quadrants = [
        { x: 0, y: 0, label: 'Low Openness\nHigh Lock-In', color: '#fee2e2' },
        { x: innerWidth / 2, y: 0, label: 'High Openness\nHigh Lock-In', color: '#fef3c7' },
        { x: 0, y: innerHeight / 2, label: 'Low Openness\nLow Lock-In', color: '#e0e7ff' },
        { x: innerWidth / 2, y: innerHeight / 2, label: 'High Openness\nLow Lock-In', color: '#d1fae5' },
      ];

      quadrants.forEach(q => {
        g.append('rect')
          .attr('x', q.x)
          .attr('y', q.y)
          .attr('width', innerWidth / 2)
          .attr('height', innerHeight / 2)
          .attr('fill', q.color)
          .attr('opacity', 0.3);
      });

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
        .call(d3.axisBottom(xScale).ticks(5));

      xAxis.selectAll('text')
        .style('font-size', '11px')
        .style('fill', '#475569');

      xAxis.select('.domain')
        .attr('stroke', '#cbd5e1')
        .attr('stroke-width', 2);

      // Y axis
      const yAxis = g.append('g')
        .call(d3.axisLeft(yScale).ticks(5));

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
        .text('Openness Score →');

      // Y axis label
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -55)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .style('fill', '#334155')
        .text('← Lock-In Index');

      // Center lines
      g.append('line')
        .attr('x1', innerWidth / 2)
        .attr('x2', innerWidth / 2)
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .attr('stroke', '#94a3b8')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '5,5');

      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerWidth)
        .attr('y1', innerHeight / 2)
        .attr('y2', innerHeight / 2)
        .attr('stroke', '#94a3b8')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '5,5');

      // Plot providers
      providers.forEach(provider => {
        const circle = g.append('circle')
          .attr('cx', xScale(provider.opennessScore))
          .attr('cy', yScale(provider.lockInIndex))
          .attr('r', 12)
          .attr('fill', provider.color)
          .attr('stroke', 'white')
          .attr('stroke-width', 2)
          .style('cursor', 'pointer')
          .style('opacity', 0.8);

        g.append('text')
          .attr('x', xScale(provider.opennessScore))
          .attr('y', yScale(provider.lockInIndex) - 20)
          .attr('text-anchor', 'middle')
          .attr('fill', '#1e293b')
          .style('font-size', '13px')
          .style('font-weight', '600')
          .text(provider.provider);

        circle.on('mouseover', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', 15)
            .style('opacity', 1);
        }).on('mouseout', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', 12)
            .style('opacity', 0.8);
        });
      });
    } else if (viewMode === 'breakdown') {
      // View 2: Factor Breakdown - Stacked bar chart
      const factorData = ecosystemData.lockInOpenness.factorBreakdown;
      const providers = Object.keys(factorData);

      const xScale = d3.scaleBand()
        .domain(providers)
        .range([0, innerWidth])
        .padding(0.4);

      const yScale = d3.scaleLinear()
        .domain([0, 10])
        .range([innerHeight, 0]);

      // Color scale for metrics - different colors for Lock-In vs Openness aspects
      const metricColors: any = {
        // Lock-In metrics (darker, more saturated)
        switchingCosts: '#dc2626',
        dataPortability_lockIn: '#ea580c',
        userControl_lockIn: '#ca8a04',
        // Openness metrics (lighter, more vibrant)
        dataPortability_openness: '#06b6d4',
        interoperability: '#3b82f6',
        userControl_openness: '#10b981'
      };

      const metricLabels: any = {
        switchingCosts: 'Switching Costs',
        dataPortability_lockIn: 'Data Portability (Absence)',
        userControl_lockIn: 'User Control (Lack)',
        dataPortability_openness: 'Data Portability (Presence)',
        interoperability: 'Interoperability',
        userControl_openness: 'User Control (Presence)'
      };

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

      // X axis with only provider names
      g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .selectAll('text')
        .data(providers)
        .join('text')
        .attr('x', d => (xScale(d) || 0) + xScale.bandwidth() / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .style('fill', '#475569')
        .text(d => d);

      // Y axis
      const yAxis = g.append('g')
        .call(d3.axisLeft(yScale).ticks(10));

      yAxis.selectAll('text')
        .style('font-size', '12px')
        .style('fill', '#475569');

      yAxis.select('.domain')
        .attr('stroke', '#cbd5e1')
        .attr('stroke-width', 2);

      // Y axis label
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -55)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .style('fill', '#334155')
        .text('Score');

      // Draw stacked bars for each provider
      providers.forEach(provider => {
        const providerData = factorData[provider as keyof typeof factorData];
        const barWidth = (xScale.bandwidth() - 10) / 2;
        const x = xScale(provider) || 0;

        // Lock-In bar (left)
        let lockInY = innerHeight;
        [
          { key: 'switchingCosts', colorKey: 'switchingCosts' },
          { key: 'dataPortability', colorKey: 'dataPortability_lockIn' },
          { key: 'userControl', colorKey: 'userControl_lockIn' }
        ].forEach(({ key, colorKey }) => {
          const value = providerData.lockIn[key as keyof typeof providerData.lockIn];
          if (value > 0) {
            const height = innerHeight - yScale(value);

            g.append('rect')
              .attr('x', x)
              .attr('y', lockInY - height)
              .attr('width', barWidth)
              .attr('height', height)
              .attr('fill', metricColors[colorKey])
              .attr('rx', 2)
              .style('opacity', 0.85)
              .append('title')
              .text(`${metricLabels[colorKey]}: ${value.toFixed(1)}`);

            lockInY -= height;
          }
        });

        // Openness bar (right)
        let opennessY = innerHeight;
        [
          { key: 'dataPortability', colorKey: 'dataPortability_openness' },
          { key: 'interoperability', colorKey: 'interoperability' },
          { key: 'userControl', colorKey: 'userControl_openness' }
        ].forEach(({ key, colorKey }) => {
          const value = providerData.openness[key as keyof typeof providerData.openness];
          if (value > 0) {
            const height = innerHeight - yScale(value);

            g.append('rect')
              .attr('x', x + barWidth + 10)
              .attr('y', opennessY - height)
              .attr('width', barWidth)
              .attr('height', height)
              .attr('fill', metricColors[colorKey])
              .attr('rx', 2)
              .style('opacity', 0.85)
              .append('title')
              .text(`${metricLabels[colorKey]}: ${value.toFixed(1)}`);

            opennessY -= height;
          }
        });

        // Category labels below bars
        g.append('text')
          .attr('x', x + barWidth / 2)
          .attr('y', innerHeight + 40)
          .attr('text-anchor', 'middle')
          .style('font-size', '11px')
          .style('fill', '#ef4444')
          .style('font-weight', '600')
          .text('Lock-In');

        g.append('text')
          .attr('x', x + barWidth * 1.5 + 10)
          .attr('y', innerHeight + 40)
          .attr('text-anchor', 'middle')
          .style('font-size', '11px')
          .style('fill', '#3b82f6')
          .style('font-weight', '600')
          .text('Openness');
      });

      // Add legend for metrics - organized by Lock-In and Openness
      const legendX = innerWidth - 220;
      const legendY = 20;

      // Lock-In section
      g.append('text')
        .attr('x', legendX)
        .attr('y', legendY)
        .style('font-size', '11px')
        .style('font-weight', '700')
        .style('fill', '#dc2626')
        .text('Lock-In Metrics:');

      ['switchingCosts', 'dataPortability_lockIn', 'userControl_lockIn'].forEach((metric, i) => {
        const y = legendY + 18 + i * 18;

        g.append('rect')
          .attr('x', legendX)
          .attr('y', y - 9)
          .attr('width', 10)
          .attr('height', 10)
          .attr('fill', metricColors[metric])
          .attr('rx', 2)
          .style('opacity', 0.85);

        g.append('text')
          .attr('x', legendX + 15)
          .attr('y', y)
          .style('font-size', '10px')
          .style('fill', '#475569')
          .text(metricLabels[metric]);
      });

      // Openness section
      const opennessStartY = legendY + 80;
      g.append('text')
        .attr('x', legendX)
        .attr('y', opennessStartY)
        .style('font-size', '11px')
        .style('font-weight', '700')
        .style('fill', '#3b82f6')
        .text('Openness Metrics:');

      ['dataPortability_openness', 'interoperability', 'userControl_openness'].forEach((metric, i) => {
        const y = opennessStartY + 18 + i * 18;

        g.append('rect')
          .attr('x', legendX)
          .attr('y', y - 9)
          .attr('width', 10)
          .attr('height', 10)
          .attr('fill', metricColors[metric])
          .attr('rx', 2)
          .style('opacity', 0.85);

        g.append('text')
          .attr('x', legendX + 15)
          .attr('y', y)
          .style('font-size', '10px')
          .style('fill', '#475569')
          .text(metricLabels[metric]);
      });

    } else if (viewMode === 'timeline') {
      // View 3: Momentum Timeline - Line chart over time
      const timeline = ecosystemData.lockInOpenness.momentumTimeline;
      const providers = Object.keys(timeline);
      const periods = timeline[providers[0] as keyof typeof timeline].map((d: any) => d.period);

      const xScale = d3.scalePoint()
        .domain(periods)
        .range([0, innerWidth])
        .padding(0.1);

      const yScale = d3.scaleLinear()
        .domain([0, 10])
        .range([innerHeight, 0]);

      const colorScale = d3.scaleOrdinal()
        .domain(providers)
        .range(['#10A37F', '#D97757', '#00A4EF', '#4285F4', '#0668E1']);

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
        .style('font-size', '10px')
        .style('fill', '#475569')
        .attr('transform', 'rotate(-15)')
        .style('text-anchor', 'end');

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

      // Y axis label
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -55)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .style('fill', '#334155')
        .text('Score');

      // Line generators
      const opennessLine = d3.line<any>()
        .x(d => xScale(d.period) || 0)
        .y(d => yScale(d.opennessScore))
        .curve(d3.curveMonotoneX);

      const lockInLine = d3.line<any>()
        .x(d => xScale(d.period) || 0)
        .y(d => yScale(d.lockInIndex))
        .curve(d3.curveMonotoneX);

      // Draw lines for each provider
      providers.forEach((provider, idx) => {
        const providerTimeline = timeline[provider as keyof typeof timeline];

        // Openness line (solid)
        g.append('path')
          .datum(providerTimeline)
          .attr('fill', 'none')
          .attr('stroke', colorScale(provider) as string)
          .attr('stroke-width', 2.5)
          .attr('d', opennessLine)
          .style('opacity', 0.8);

        // Lock-In line (dashed)
        g.append('path')
          .datum(providerTimeline)
          .attr('fill', 'none')
          .attr('stroke', colorScale(provider) as string)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5')
          .attr('d', lockInLine)
          .style('opacity', 0.6);

        // Add circles for data points
        providerTimeline.forEach((point: any) => {
          // Openness point
          g.append('circle')
            .attr('cx', xScale(point.period) || 0)
            .attr('cy', yScale(point.opennessScore))
            .attr('r', 4)
            .attr('fill', colorScale(provider) as string)
            .attr('stroke', 'white')
            .attr('stroke-width', 2);

          // Lock-In point
          g.append('circle')
            .attr('cx', xScale(point.period) || 0)
            .attr('cy', yScale(point.lockInIndex))
            .attr('r', 4)
            .attr('fill', colorScale(provider) as string)
            .attr('stroke', 'white')
            .attr('stroke-width', 1)
            .style('opacity', 0.6);
        });
      });

      // Legend for line styles
      g.append('text')
        .attr('x', innerWidth - 10)
        .attr('y', 20)
        .attr('text-anchor', 'end')
        .style('font-size', '11px')
        .style('fill', '#475569')
        .text('─── Openness  - - - Lock-In');
    }

  }, [viewMode, dimensions]);

  return (
    <div className="flex flex-col w-full h-full">
      {/* Legend Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <div className="flex items-center gap-4">
          <div className="text-sm font-medium text-slate-700">
            {viewMode === 'matrix'
              ? 'Strategy Matrix: Openness vs Lock-In'
              : viewMode === 'breakdown'
              ? 'Factor Breakdown by Provider'
              : 'Momentum Timeline (2023-2026)'}
          </div>

          {/* View Toggle */}
          <div className="flex gap-2">
            <button
              onClick={() => onViewModeChange?.('matrix')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                viewMode === 'matrix'
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Matrix
            </button>
            <button
              onClick={() => onViewModeChange?.('breakdown')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                viewMode === 'breakdown'
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Breakdown
            </button>
            <button
              onClick={() => onViewModeChange?.('timeline')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                viewMode === 'timeline'
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Timeline
            </button>
          </div>
        </div>

        {/* Line Color Legend */}
        <div className="flex gap-3 text-xs">
          {ecosystemData.lockInOpenness.strategyMatrix.map((provider) => (
            <div key={provider.provider} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: provider.color }}
              ></div>
              <span className="text-slate-700">{provider.provider}</span>
            </div>
          ))}
        </div>
      </div>


      {/* Chart */}
      <div ref={containerRef} className="flex-1 w-full min-h-[700px]">
        <svg ref={svgRef}></svg>
      </div>
    </div>
  );
}
