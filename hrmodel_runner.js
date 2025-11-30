// HR Model App for Bangle.js
// Shows raw vs corrected HR with data logging
//============================================================================

// Import model data
const modelData = require('hr_model_10feat_2.1mae.js'); // <-- Change this to your model file

//============================================================================
// HR MODEL RUNNER CLASS
//============================================================================

class HRModelRunner {
  constructor(modelData) {
    this.coefficients = modelData.coefficients;
    this.intercept = modelData.intercept;
    this.scalerMean = modelData.scalerMean;
    this.scalerScale = modelData.scalerScale;
    this.features = modelData.features;
    this.bufferSize = modelData.bufferSize;
    this.metadata = modelData.metadata || {};
    this.hrBuffer = [];
    
    if (!this.coefficients || !this.features || this.coefficients.length !== this.features.length) {
      throw new Error("Invalid model data");
    }
  }
  
  addReading(hr) {
    if (hr <= 0 || hr > 220) return;
    this.hrBuffer.push(hr);
    if (this.hrBuffer.length > this.bufferSize) {
      this.hrBuffer.shift();
    }
  }
  
  isReady() {
    return this.hrBuffer.length >= this.bufferSize;
  }
  
  computeFeatures() {
    if (this.hrBuffer.length < this.bufferSize) return null;
    
    const features = [];
    const n = this.hrBuffer.length;
    
    for (let feature of this.features) {
      let value = null;
      
      if (feature.startsWith('hr_lag_')) {
        const lag = parseInt(feature.split('_')[2]);
        value = this.hrBuffer[n - lag - 1];
      }
      else if (feature.startsWith('hr_change_')) {
        const lag = parseInt(feature.split('_')[2]);
        value = this.hrBuffer[n - 1] - this.hrBuffer[n - lag - 1];
      }
      else if (feature.startsWith('hr_accel_')) {
        const lag = parseInt(feature.split('_')[2]);
        const change1 = this.hrBuffer[n - 1] - this.hrBuffer[n - 2];
        const change2 = this.hrBuffer[n - lag - 1] - this.hrBuffer[n - lag - 2];
        value = change1 - change2;
      }
      else if (feature.startsWith('hr_rolling_mean_')) {
        const window = parseInt(feature.split('_')[3]);
        const start = n - window;
        let sum = 0;
        for (let j = start; j < n; j++) sum += this.hrBuffer[j];
        value = sum / window;
      }
      else if (feature.startsWith('hr_rolling_std_')) {
        const window = parseInt(feature.split('_')[3]);
        const start = n - window;
        let sum = 0;
        for (let j = start; j < n; j++) sum += this.hrBuffer[j];
        const mean = sum / window;
        let variance = 0;
        for (let j = start; j < n; j++) {
          variance += Math.pow(this.hrBuffer[j] - mean, 2);
        }
        value = Math.sqrt(variance / window);
      }
      else if (feature.startsWith('hr_rolling_min_')) {
        const window = parseInt(feature.split('_')[3]);
        const start = n - window;
        let min = this.hrBuffer[start];
        for (let j = start + 1; j < n; j++) {
          if (this.hrBuffer[j] < min) min = this.hrBuffer[j];
        }
        value = min;
      }
      else if (feature.startsWith('hr_rolling_max_')) {
        const window = parseInt(feature.split('_')[3]);
        const start = n - window;
        let max = this.hrBuffer[start];
        for (let j = start + 1; j < n; j++) {
          if (this.hrBuffer[j] > max) max = this.hrBuffer[j];
        }
        value = max;
      }
      else {
        value = 0;
      }
      
      features.push(value);
    }
    
    return features;
  }
  
  standardize(features) {
    const standardized = [];
    for (let i = 0; i < features.length; i++) {
      standardized[i] = (features[i] - this.scalerMean[i]) / this.scalerScale[i];
    }
    return standardized;
  }
  
  predict() {
    const features = this.computeFeatures();
    if (!features) return null;
    
    const standardized = this.standardize(features);
    let prediction = this.intercept;
    for (let i = 0; i < standardized.length; i++) {
      prediction += this.coefficients[i] * standardized[i];
    }
    
    return Math.round(prediction);
  }
  
  getCorrectedHR(rawHR) {
    this.addReading(rawHR);
    const predicted = this.predict();
    return predicted !== null ? predicted : rawHR;
  }
  
  getModelInfo() {
    return {
      features: this.features.length,
      bufferSize: this.bufferSize,
      mae: this.metadata.mae_bpm || "unknown",
      bufferFill: this.hrBuffer.length,
      isReady: this.isReady()
    };
  }
}

//============================================================================
// APP CODE
//============================================================================

// Initialize model
const hrModel = new HRModelRunner(modelData);
let rawHR = 0;
let correctedHR = 0;
let isLogging = false;
let logData = [];
let logStartTime = null;

// File for logging
const LOG_FILE = "hrmodel.log";

// UI Configuration
const COLORS = {
  bg: "#000",
  raw: "#888",
  corrected: "#0f0",
  text: "#888",
  label: "#888",
  button: "#444",
  buttonText: "#888",
  logOn: "#0f0",
  logOff: "#f00"
};

let showingWelcome = true;

// Draw welcome screen
function drawWelcome() {
  g.clear();
  g.setColor(COLORS.logOff);
  
  g.setFont("6x8", 2.5);
  g.drawString("<3 Fixer ML", 10, 30);
  
  g.setFont("6x8", 2);
  g.drawString("Model Ready!", 10, 80);
  
  const modelInfo = hrModel.getModelInfo();
  g.setFont("6x8", 1.5);
  g.drawString("MAE: " + modelInfo.mae + " bpm", 10, 110);
  g.drawString("Features: " + modelInfo.features, 10, 130);
  
  g.setFont("6x8", 2);
  g.drawString("Tap to enter", 10, 160);
  g.drawString("Button = Log", 10, 180);
  
  g.flip();
}

// Draw the main UI
function drawUI() {
  g.clear();
  
  const GAP = 30;
  const bigtext = 3;
  const liltext = 2;
  let y = 10;
  
  // BIG TEXT
  g.setFont("6x8", bigtext);
  
  // Corrected HR
  g.setColor(COLORS.corrected);
  g.drawString("MLc " + (correctedHR > 0 ? correctedHR : "--"), 10, y);
  y += GAP;
  
  // Raw HR
  g.setColor(COLORS.raw);
  g.drawString("Raw " + (rawHR > 0 ? rawHR : "--"), 10, y);
  y += 1.5*GAP;
  
  // lil text
  g.setFont("6x8", liltext);
  
  // Buffer
  const modelInfo = hrModel.getModelInfo();
  g.setColor(COLORS.raw);
  g.drawString("Buffer " + modelInfo.bufferFill + "/" + modelInfo.bufferSize, 10, y);
  y += GAP;
  
  // Sample count
  g.drawString("Samples " + logData.length, 10, y);
  y += GAP;
  
  // Log status dot
  g.setColor(isLogging ? COLORS.logOn : COLORS.logOff);
  g.fillCircle(20, y + 10, 10);
  g.setColor(COLORS.text);
  g.drawString(isLogging ? "LOGGING" : "Stopped", 40, y);
  
  g.flip();
}

// Start/Stop logging
function toggleLogging() {
  isLogging = !isLogging;
  
  if (isLogging) {
    logData = [];
    logStartTime = Date.now();
    console.log("Logging started");
  } else {
    console.log("Logging stopped. Saving...");
    saveLog();
  }
  
  drawUI();
}

// Save log to storage
function saveLog() {
  if (logData.length === 0) {
    console.log("No data to save");
    return;
  }
  
  const logContent = logData.map(d => 
    d.time + "," + d.raw + "," + d.corrected + "," + d.diff
  ).join("\n");
  
  const header = "timestamp,raw_hr,corrected_hr,difference\n";
  require("Storage").write(LOG_FILE, header + logContent);
  
  console.log("Saved " + logData.length + " samples to " + LOG_FILE);
}

// HR monitor callback
function onHRM(hrm) {
  rawHR = hrm.bpm;
  
  if (rawHR > 0) {
    correctedHR = hrModel.getCorrectedHR(rawHR);
    
    if (isLogging && correctedHR > 0) {
      const diff = correctedHR - rawHR;
      logData.push({
        time: Date.now() - logStartTime,
        raw: rawHR,
        corrected: correctedHR,
        diff: diff
      });
    }
  }
  
  if (!showingWelcome) {
    drawUI();
  }
}

// Button handler
setWatch(toggleLogging, BTN1, {repeat: true, edge: "rising"});

// Touch handler
Bangle.on('touch', function() {
  if (showingWelcome) {
    showingWelcome = false;
    drawUI();
  }
});

// Start HR monitoring
Bangle.setHRMPower(1);
Bangle.on('HRM', onHRM);

// Show welcome screen
drawWelcome();

// Model info
const modelInfo = hrModel.getModelInfo();
console.log("HR Model loaded:");
console.log("  MAE: " + modelInfo.mae + " bpm");
console.log("  Features: " + modelInfo.features);
console.log("  Buffer: " + modelInfo.bufferSize + " readings");