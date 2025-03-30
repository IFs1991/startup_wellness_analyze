import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { scaleBand, scaleLinear } from 'd3-scale';
import { axisBottom, axisLeft } from 'd3-axis';
import { area, curveCatmullRom } from 'd3-shape';
import { schemeCategory10 } from 'd3-scale-chromatic';
import { mean, median, max } from 'd3-array';

interface ViolinData {
  name: string;
  values: number[];
}

interface ViolinChartProps {
  data: ViolinData[];
  width?: number;
  height?: number;
  marginTop?: number;
  marginRight?: number;
  marginBottom?: number;
  marginLeft?: number;
  xDomain?: string[];
  yDomain?: [number, number];
  thresholds?: number;
  colors?: string[];
}

// 密度データの型定義
type DensityPoint = [number, number];

export const ViolinChart: React.FC<ViolinChartProps> = ({
  data,
  width = 800,
  height = 400,
  marginTop = 40,
  marginRight = 30,
  marginBottom = 40,
  marginLeft = 40,
  xDomain,
  yDomain,
  thresholds = 20,
  colors = schemeCategory10
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data || data.length === 0) return;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    // Compute domains
    const keys = xDomain || data.map(d => d.name);
    const allValues = data.flatMap(d => d.values);
    const y = yDomain || [
      Math.min(...allValues),
      Math.max(...allValues)
    ];

    // Set up scales
    const x = scaleBand()
      .domain(keys)
      .range([marginLeft, width - marginRight])
      .padding(0.15);

    const yScale = scaleLinear()
      .domain(y)
      .range([height - marginBottom, marginTop]);

    // Create violin shapes
    const violinWidth = Math.min(80, x.bandwidth());

    // Set up the SVG container
    const svg = d3.select(svgRef.current);

    // Add X and Y axes
    svg.append('g')
      .attr('transform', `translate(0,${height - marginBottom})`)
      .call(axisBottom(x) as any);

    svg.append('g')
      .attr('transform', `translate(${marginLeft},0)`)
      .call(axisLeft(yScale) as any);

    // Create violin plots
    data.forEach((violinData, i) => {
      // Compute kernel density estimation
      const kde = kernelDensityEstimator(kernelEpanechnikov(7),
        yScale.ticks(thresholds));
      const density = kde(violinData.values);

      // Calculate the maximal width
      const maxWidth = max(density, (d: DensityPoint) => d[1]) || 0;

      // Scale for the violin width
      const xNum = scaleLinear()
        .domain([0, maxWidth])
        .range([0, violinWidth / 2]);

      // Draw the violin path
      const violinG = svg.append('g')
        .attr('transform', `translate(${x(violinData.name)! + x.bandwidth() / 2},0)`);

      // Create the outline
      const areaGenerator = area<DensityPoint>()
        .x0((d: DensityPoint) => -xNum(d[1]))
        .x1((d: DensityPoint) => xNum(d[1]))
        .y((d: DensityPoint) => yScale(d[0]))
        .curve(curveCatmullRom);

      // 文字列型を確保してnullを排除
      const pathData = areaGenerator(density) || '';

      violinG.append('path')
        .attr('d', pathData)
        .style('fill', colors[i % colors.length])
        .style('opacity', 0.8)
        .style('stroke', 'none');

      // Add median line
      const medianValue = median(violinData.values) || 0;
      violinG.append('line')
        .attr('x1', -violinWidth / 2)
        .attr('x2', violinWidth / 2)
        .attr('y1', yScale(medianValue))
        .attr('y2', yScale(medianValue))
        .style('stroke', 'black')
        .style('stroke-width', 2);
    });

    // Helper functions for KDE
    function kernelDensityEstimator(kernel: (v: number) => number, X: number[]) {
      return function(V: number[]): DensityPoint[] {
        return X.map(x => [x, mean(V, v => kernel(x - v)) || 0]);
      };
    }

    function kernelEpanechnikov(k: number) {
      return function(v: number): number {
        return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
      };
    }
  }, [data, width, height, marginTop, marginRight, marginBottom, marginLeft,
      xDomain, yDomain, thresholds, colors]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ display: 'block', overflow: 'visible' }}
    />
  );
};