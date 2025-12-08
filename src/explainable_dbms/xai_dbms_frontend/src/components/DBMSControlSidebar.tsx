import { useState } from "react";
import { Upload, X } from "lucide-react";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

interface DBMSControlSidebarProps {
  onRunAnalysis: (params: AnalysisParams) => void;
  isAnalyzing: boolean;
}

export interface AnalysisParams {
  csvFile?: File;
  targetColumn: string;
}

export function DBMSControlSidebar({ onRunAnalysis, isAnalyzing }: DBMSControlSidebarProps) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [analysisError, setAnalysisError] = useState(false);


  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.csv')) {
        setUploadedFile(file);
        setAnalysisError(false);
        uploadFileToBackend(file);
      }
    }
  };

  const uploadFileToBackend = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload file');
      }

      console.log('File uploaded successfully');
    } catch (error) {
      console.error('Upload error:', error);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadedFile(file);
      setAnalysisError(false);
      uploadFileToBackend(file);
    }
  };

  const handleRunAnalysis = () => {
    if (!uploadedFile) {
      setAnalysisError(true);
      return;
    }

    setAnalysisError(false);
    // Pass empty target column - LLM will suggest it
    onRunAnalysis({
      csvFile: uploadedFile,
      targetColumn: "",
    });
  };

  return (
    <div className="w-80 border-r border-border-color bg-bg-secondary p-6 overflow-y-auto space-y-6 animate-slide-in-left">
      {/* Upload CSV Section */}
      <div className="space-y-3 animate-fade-in" style={{ animationDelay: '0.1s', opacity: 0 }}>
        <div className="flex items-center gap-2 mb-3 group">
          <Upload className="w-4 h-4 transition-transform duration-300 group-hover:scale-110" style={{ color: 'var(--text-primary)' }} />
          <Label style={{ color: 'var(--text-primary)' }}>Upload CSV</Label>
        </div>

        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`
            relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 transform hover:scale-[1.02]
            ${dragActive
              ? 'border-accent-primary bg-accent-primary/10 scale-[1.02]'
              : 'border-border-color hover:border-accent-primary/50 hover:shadow-lg'
            }
          `}
        >
          <input
            type="file"
            id="csv-upload"
            className="hidden"
            accept=".csv"
            onChange={handleFileInput}
          />
          <label htmlFor="csv-upload" className="cursor-pointer group">
            <Upload
              className="w-8 h-8 mx-auto mb-2 transition-all duration-300 group-hover:scale-110 group-hover:-translate-y-1"
              style={{ color: dragActive ? 'var(--accent-primary)' : 'var(--text-secondary)' }}
            />
            {uploadedFile ? (
              <div>
                <p className="mb-1" style={{ color: 'var(--text-primary)' }}>
                  {uploadedFile.name}
                </p>
                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  Click to change file
                </p>
              </div>
            ) : (
              <div>
                <p className="mb-1" style={{ color: 'var(--text-primary)' }}>
                  Drop File Here
                </p>
                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  or
                </p>
                <p className="text-xs" style={{ color: 'var(--text-primary)' }}>
                  Click to Upload
                </p>
              </div>
            )}
          </label>
        </div>
      </div>

      {/* Run Analysis Button */}
      <Button
        onClick={handleRunAnalysis}
        disabled={isAnalyzing || !uploadedFile}
        className="w-full bg-text-secondary hover:bg-text-secondary/90 text-bg-primary transform hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 animate-fade-in shadow-lg hover:shadow-xl"
        style={{ animationDelay: '0.3s', opacity: 0 }}
      >
        {isAnalyzing ? (
          <span className="flex items-center gap-2">
            <span className="w-4 h-4 border-2 border-bg-primary border-t-transparent rounded-full animate-spin"></span>
            Analyzing...
          </span>
        ) : (
          'Run Analysis'
        )}
      </Button>

      {/* Analysis Status */}
      <div className="space-y-3 bg-bg-primary rounded-lg p-4 border border-border-color hover:border-accent-primary/30 transition-all duration-300 animate-fade-in" style={{ animationDelay: '0.4s', opacity: 0 }}>
        <div className="flex items-center justify-between">
          <Label style={{ color: 'var(--text-primary)' }}>
            Analysis Status
          </Label>
          <X className="w-4 h-4 cursor-pointer hover:text-accent-primary transition-colors duration-200" style={{ color: 'var(--text-secondary)' }} />
        </div>

        <div className="py-4 text-center">
          {analysisError ? (
            <p className="text-xs px-3 py-2 rounded-full bg-red-500/20 border border-red-500 inline-block animate-scale-in" style={{ color: '#ef4444' }}>
              Error
            </p>
          ) : isAnalyzing ? (
            <p className="text-xs px-3 py-2 rounded-full bg-accent-primary/20 border border-accent-primary inline-block animate-pulse-glow" style={{ color: 'var(--accent-primary)' }}>
              Analyzing...
            </p>
          ) : (
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Ready to analyze
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
