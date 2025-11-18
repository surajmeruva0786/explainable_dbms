import { useState } from "react";
import { Upload, Loader2 } from "lucide-react";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Slider } from "./ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Textarea } from "./ui/textarea";

interface ControlSidebarProps {
  onGenerate: (params: GenerationParams) => void;
  isGenerating: boolean;
}

export interface GenerationParams {
  prompt: string;
  model: string;
  steps: number;
  guidance: number;
  imageFile?: File;
}

export function ControlSidebar({ onGenerate, isGenerating }: ControlSidebarProps) {
  const [prompt, setPrompt] = useState("A futuristic cityscape at night with neon lights");
  const [model, setModel] = useState("stable-diffusion-xl");
  const [steps, setSteps] = useState([50]);
  const [guidance, setGuidance] = useState([7.5]);
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

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
      setUploadedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadedFile(e.target.files[0]);
    }
  };

  const handleGenerate = () => {
    onGenerate({
      prompt,
      model,
      steps: steps[0],
      guidance: guidance[0],
      imageFile: uploadedFile || undefined,
    });
  };

  return (
    <div className="w-80 border-r border-border-color bg-bg-secondary p-6 overflow-y-auto">
      <div className="space-y-6">
        <div>
          <h2 className="mb-4" style={{ color: 'var(--text-primary)' }}>Generation Controls</h2>
          
          <div className="space-y-4">
            {/* Prompt Input */}
            <div className="space-y-2">
              <Label htmlFor="prompt" style={{ color: 'var(--text-primary)' }}>
                Prompt
              </Label>
              <Textarea
                id="prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe what you want to generate..."
                className="min-h-[100px] bg-bg-primary border-border-color focus:border-accent-primary focus:ring-2 focus:ring-accent-primary/20 resize-none"
                style={{ color: 'var(--text-primary)' }}
              />
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label htmlFor="model" style={{ color: 'var(--text-primary)' }}>
                Model
              </Label>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger 
                  id="model"
                  className="bg-bg-primary border-border-color focus:border-accent-primary focus:ring-2 focus:ring-accent-primary/20"
                  style={{ color: 'var(--text-primary)' }}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-bg-secondary border-border-color">
                  <SelectItem value="stable-diffusion-xl" className="focus:bg-accent-primary/20">
                    Stable Diffusion XL
                  </SelectItem>
                  <SelectItem value="midjourney-v6" className="focus:bg-accent-primary/20">
                    Midjourney v6
                  </SelectItem>
                  <SelectItem value="dalle-3" className="focus:bg-accent-primary/20">
                    DALL-E 3
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Steps Slider */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label style={{ color: 'var(--text-primary)' }}>Steps</Label>
                <span className="text-sm px-2 py-1 rounded bg-bg-primary" style={{ color: 'var(--accent-primary)' }}>
                  {steps[0]}
                </span>
              </div>
              <Slider
                value={steps}
                onValueChange={setSteps}
                min={1}
                max={100}
                step={1}
                className="w-full"
              />
            </div>

            {/* Guidance Scale Slider */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label style={{ color: 'var(--text-primary)' }}>Guidance Scale</Label>
                <span className="text-sm px-2 py-1 rounded bg-bg-primary" style={{ color: 'var(--accent-primary)' }}>
                  {guidance[0].toFixed(1)}
                </span>
              </div>
              <Slider
                value={guidance}
                onValueChange={setGuidance}
                min={1}
                max={20}
                step={0.5}
                className="w-full"
              />
            </div>

            {/* File Upload Zone */}
            <div className="space-y-2">
              <Label style={{ color: 'var(--text-primary)' }}>
                Upload Reference Image (Optional)
              </Label>
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`
                  relative border-2 border-dashed rounded-lg p-8 text-center transition-all
                  ${dragActive 
                    ? 'border-accent-primary bg-accent-primary/10' 
                    : 'border-border-color hover:border-accent-primary/50'
                  }
                `}
              >
                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  accept="image/*"
                  onChange={handleFileInput}
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload 
                    className="w-10 h-10 mx-auto mb-3" 
                    style={{ color: dragActive ? 'var(--accent-primary)' : 'var(--text-secondary)' }} 
                  />
                  <p className="mb-1" style={{ color: 'var(--text-primary)' }}>
                    {uploadedFile ? uploadedFile.name : 'Drop image here or click to upload'}
                  </p>
                  <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    PNG, JPG up to 10MB
                  </p>
                </label>
              </div>
            </div>

            {/* Generate Button */}
            <Button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="w-full bg-accent-primary hover:bg-accent-primary/90 text-bg-primary transition-all duration-200"
              style={isGenerating ? {} : { boxShadow: '0 0 20px rgba(0, 242, 234, 0.3)' }}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                'Generate'
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
