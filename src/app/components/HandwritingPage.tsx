import { useState } from "react";
import { motion } from "motion/react";
import { Upload, FileText, PenTool } from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { Label } from "./ui/label";

export function HandwritingPage() {
  const [worksheetImage, setWorksheetImage] = useState<File | null>(null);
  const [analyzedStyle, setAnalyzedStyle] = useState<any>(null);
  const [textFields, setTextFields] = useState<{ [key: string]: string }>({});
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const apiBase = "http://localhost:8000";

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setWorksheetImage(file);
      setAnalyzedStyle(null);
      setGeneratedImage(null);
    }
  };

  const analyzeHandwriting = async () => {
    if (!worksheetImage) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('image', worksheetImage);

    try {
      const response = await fetch(`${apiBase}/analyze-handwriting`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      if (result.style) {
        setAnalyzedStyle(result.style);
        // Initialize text fields based on detected fields
        const fields: { [key: string]: string } = {};
        result.style.fields?.forEach((field: string) => {
          fields[field] = '';
        });
        setTextFields(fields);
      }
    } catch (error) {
      console.error('Analysis failed:', error);
    }
    setLoading(false);
  };

  const generateHandwriting = () => {
    if (!worksheetImage || !analyzedStyle) return;

    // Mock generation: in real implementation, draw on canvas with style
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Load the worksheet image
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Draw text overlays with simulated handwriting style
      ctx.font = `${analyzedStyle.pressure * 20}px serif`;
      ctx.fillStyle = 'black';
      ctx.save();
      ctx.rotate((analyzedStyle.slant || 0) * Math.PI / 180);

      // Example: draw sample text
      Object.entries(textFields).forEach(([field, text], index) => {
        if (text) {
          ctx.fillText(text, 50, 100 + index * 50);
        }
      });

      ctx.restore();

      setGeneratedImage(canvas.toDataURL());
    };
    img.src = URL.createObjectURL(worksheetImage);
  };

  return (
    <div className="p-8 lg:pt-8 pt-20 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-white mb-1">Handwriting Worksheet Filler</h1>
          <p className="text-sm text-zinc-400">Analyze worksheet layout and overlay custom handwritten text</p>
        </div>
      </div>

      {/* Upload Section */}
      <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
        <div className="space-y-4">
          <div>
            <Label htmlFor="worksheet-upload" className="text-white">Upload Worksheet Image</Label>
            <div className="mt-2">
              <input
                id="worksheet-upload"
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
              <label
                htmlFor="worksheet-upload"
                className="flex items-center justify-center w-full h-32 border-2 border-dashed border-zinc-600 rounded-lg cursor-pointer hover:border-zinc-400 transition-colors"
              >
                <div className="text-center">
                  <Upload className="mx-auto h-8 w-8 text-zinc-400 mb-2" />
                  <p className="text-sm text-zinc-400">
                    {worksheetImage ? worksheetImage.name : 'Click to upload worksheet image'}
                  </p>
                </div>
              </label>
            </div>
          </div>

          {worksheetImage && (
            <Button onClick={analyzeHandwriting} disabled={loading} className="w-full">
              <PenTool className="w-4 h-4 mr-2" />
              {loading ? 'Analyzing...' : 'Analyze Handwriting Style'}
            </Button>
          )}
        </div>
      </Card>

      {/* Analysis Results */}
      {analyzedStyle && (
        <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
          <h3 className="text-lg font-semibold text-white mb-4">Detected Handwriting Style</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <Label className="text-zinc-400">Slant</Label>
              <p className="text-white">{analyzedStyle.slant}°</p>
            </div>
            <div>
              <Label className="text-zinc-400">Pressure</Label>
              <p className="text-white">{analyzedStyle.pressure}</p>
            </div>
            <div>
              <Label className="text-zinc-400">Noise</Label>
              <p className="text-white">{analyzedStyle.noise}</p>
            </div>
            <div>
              <Label className="text-zinc-400">Fields Detected</Label>
              <p className="text-white">{analyzedStyle.fields?.length || 0}</p>
            </div>
          </div>
        </Card>
      )}

      {/* Text Input Fields */}
      {analyzedStyle && (
        <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
          <h3 className="text-lg font-semibold text-white mb-4">Fill Text Fields</h3>
          <div className="space-y-4">
            {Object.keys(textFields).map((field) => (
              <div key={field}>
                <Label htmlFor={field} className="text-zinc-400">{field}</Label>
                <Input
                  id={field}
                  value={textFields[field]}
                  onChange={(e) => setTextFields({ ...textFields, [field]: e.target.value })}
                  placeholder={`Enter text for ${field}`}
                  className="mt-1"
                />
              </div>
            ))}
            <Button onClick={generateHandwriting} className="w-full">
              <FileText className="w-4 h-4 mr-2" />
              Generate Handwritten Worksheet
            </Button>
          </div>
        </Card>
      )}

      {/* Generated Image */}
      {generatedImage && (
        <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
          <h3 className="text-lg font-semibold text-white mb-4">Generated Worksheet</h3>
          <img src={generatedImage} alt="Generated worksheet" className="max-w-full h-auto" />
          <Button className="mt-4 w-full" variant="outline">
            Download
          </Button>
        </Card>
      )}
    </div>
  );
}
