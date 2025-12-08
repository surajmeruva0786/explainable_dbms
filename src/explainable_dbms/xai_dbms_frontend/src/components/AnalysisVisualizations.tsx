import React from 'react';
import { X, BarChart3, ImageIcon, AlertCircle } from "lucide-react";

interface AnalysisResult {
  message: string;
  analysis_id: string;
  plots: Record<string, string>;
}

interface AnalysisVisualizationsProps {
  hasData: boolean;
  showError?: boolean;
  errorMessage?: string;
  analysisResult?: AnalysisResult | null;
}

export function AnalysisVisualizations({ hasData, showError, errorMessage, analysisResult }: AnalysisVisualizationsProps) {
  const PlotContainer = ({ children, title, icon: Icon }: { children: React.ReactNode; title: string; icon: any }) => (
    <div className="bg-bg-primary rounded-lg border border-border-color p-4 hover:border-accent-primary/40 transition-all duration-300 transform hover:scale-[1.01] hover:shadow-lg group flex flex-col h-full min-h-[400px]">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 transition-all duration-300 group-hover:scale-110 group-hover:rotate-12" style={{ color: 'var(--accent-primary)' }} />
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{title}</h3>
        </div>
      </div>
      <div className="flex-1 flex items-center justify-center overflow-hidden rounded-md bg-bg-secondary/30 relative">
        {showError ? (
          <div className="flex flex-col items-center gap-2 p-4 text-center">
            <AlertCircle className="w-8 h-8 text-red-500" />
            <p className="text-sm text-red-500">{errorMessage || "An error occurred"}</p>
          </div>
        ) : !hasData ? (
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            No visualization data available
          </p>
        ) : (
          children
        )}
      </div>
    </div>
  );

  const getTitleFromKey = (key: string): string => {
    const titles: Record<string, string> = {
      shap_summary: "SHAP Feature Importance Specification",
      lime_explanation: "LIME Instance Explanations",
      feature_importance: "Global Feature Importance",
      confusion_matrix: "Confusion Matrix",
      metrics: "Model Metrics",
      temp_explanation: "Query Explanation"
    };
    return titles[key] || key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  // Safe check for plots
  const plots = analysisResult?.plots || {};
  // Filter out non-image files (like json) and sort
  const plotKeys = Object.keys(plots).filter(key => {
    const url = plots[key].toLowerCase();
    return url.endsWith('.png') || url.endsWith('.jpg') || url.endsWith('.jpeg') || url.endsWith('.svg');
  }).sort();

  return (
    <div className="flex-1 p-6 space-y-4 animate-slide-in-right overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="animate-fade-in text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
          Analysis Visualizations
        </h2>
      </div>

      {showError && (
        <div className="p-4 rounded-lg border border-red-500/50 bg-red-500/10 text-red-500 mb-4 flex items-center gap-2">
          <AlertCircle className="w-5 h-5" />
          <p>{errorMessage}</p>
        </div>
      )}

      {!hasData && !showError && (
        <div className="flex flex-col items-center justify-center h-[400px] border-2 border-dashed border-border-color rounded-xl">
          <BarChart3 className="w-12 h-12 text-muted-foreground mb-4 opacity-50" />
          <p className="text-muted-foreground">Run an analysis to see visualizations</p>
        </div>
      )}

      {hasData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in" style={{ animationDelay: '0.2s' }}>
          {plotKeys.length > 0 ? (
            plotKeys.map((key) => (
              <PlotContainer key={key} title={getTitleFromKey(key)} icon={ImageIcon}>
                <img
                  src={plots[key]}
                  alt={key}
                  className="w-full h-full object-contain hover:scale-105 transition-transform duration-500"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.style.display = 'none';
                    target.parentElement!.innerText = 'Failed to load image';
                  }}
                />
              </PlotContainer>
            ))
          ) : (
            <div className="col-span-full text-center py-10 text-muted-foreground">
              No plots generated for this analysis.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
