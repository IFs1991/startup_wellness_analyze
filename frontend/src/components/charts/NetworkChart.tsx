import React, { useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

interface NetworkNode {
  id: string;
  name?: string;
  val?: number;
  color?: string;
  [key: string]: any;
}

interface NetworkLink {
  source: string;
  target: string;
  value?: number;
  color?: string;
  [key: string]: any;
}

interface NetworkChartProps {
  nodes: NetworkNode[];
  links: NetworkLink[];
  width?: number;
  height?: number;
  nodeRelSize?: number;
  nodeAutoColorBy?: string;
  linkWidth?: number | ((link: NetworkLink) => number);
  linkColor?: string | ((link: NetworkLink) => string);
  backgroundColor?: string;
}

export const NetworkChart: React.FC<NetworkChartProps> = ({
  nodes,
  links,
  width = 800,
  height = 600,
  nodeRelSize = 6,
  nodeAutoColorBy = 'group',
  linkWidth = 1.5,
  linkColor = 'rgba(50, 50, 50, 0.4)',
  backgroundColor = '#fff'
}) => {
  const graphRef = useRef();

  // Node click handler with zoom
  const handleNodeClick = useCallback((node) => {
    if (graphRef.current) {
      // @ts-ignore - Type issues with the ref
      const fg = graphRef.current;
      // Aim at node from outside
      const distance = 100;
      const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z || 0);

      fg.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: (node.z || 0) * distRatio },
        node,
        1000
      );
    }
  }, []);

  return (
    <ForceGraph2D
      ref={graphRef}
      width={width}
      height={height}
      graphData={{ nodes, links }}
      nodeRelSize={nodeRelSize}
      nodeAutoColorBy={nodeAutoColorBy}
      nodeLabel={(node) => node.name || node.id}
      onNodeClick={handleNodeClick}
      linkWidth={linkWidth}
      linkColor={linkColor}
      backgroundColor={backgroundColor}
      cooldownTicks={100}
      linkDirectionalParticles={2}
      linkDirectionalParticleWidth={2}
    />
  );
};