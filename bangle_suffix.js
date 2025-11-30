// DON'T USE THIS CODE ON ITS OWN :) 
// IT'S MEANT TO BE PASTED UNDER MODEL COEFFICIENTS BY THE NOTEBOOK
  addReading(hr) {
    this.hrBuffer.push(hr);
    if (this.hrBuffer.length > this.bufferSize) {
      this.hrBuffer.shift();
    }
  }
  
  computeFeatures() {
    if (this.hrBuffer.length < this.bufferSize) return null;
    
    const features = [];
    const n = this.hrBuffer.length;
    
    for (let feature of this.features) {
      if (feature.startsWith('hr_lag_')) {
        const lag = parseInt(feature.split('_')[2]);
        features.push(this.hrBuffer[n - lag - 1]);
      }
      else if (feature.startsWith('hr_change_')) {
        const lag = parseInt(feature.split('_')[2]);
        features.push(this.hrBuffer[n - 1] - this.hrBuffer[n - lag - 1]);
      }
      else if (feature.startsWith('hr_accel_')) {
        const change1 = this.hrBuffer[n - 1] - this.hrBuffer[n - 2];
        const change2 = this.hrBuffer[n - 2] - this.hrBuffer[n - 3];
        features.push(change1 - change2);
      }
      else if (feature.startsWith('hr_rolling_mean_')) {
        const window = parseInt(feature.split('_')[3]);
        const start = n - window;
        let sum = 0;
        for (let j = start; j < n; j++) sum += this.hrBuffer[j];
        features.push(sum / window);
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
        features.push(Math.sqrt(variance / window));
      }
      else if (feature.startsWith('hr_rolling_min_')) {
        const window = parseInt(feature.split('_')[3]);
        const start = n - window;
        let min = this.hrBuffer[start];
        for (let j = start + 1; j < n; j++) {
          if (this.hrBuffer[j] < min) min = this.hrBuffer[j];
        }
        features.push(min);
      }
      else if (feature.startsWith('hr_rolling_max_')) {
        const window = parseInt(feature.split('_')[3]);
        const start = n - window;
        let max = this.hrBuffer[start];
        for (let j = start + 1; j < n; j++) {
          if (this.hrBuffer[j] > max) max = this.hrBuffer[j];
        }
        features.push(max);
      }
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
}

// ============================================================================
// APP CODE
// ============================================================================

// Initialize
const hrModel = new HRDynamicsModel();
let rawHR = 0;
let correctedHR = 0;
let confidence = 0;
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
  label: "#888",  // Light gray for labels
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
  
  g.setFont("6x8", 2);
  g.drawString("Tap to enter", 10, 130);
  g.drawString("Button = Log", 10, 150);
  
  g.flip();
}

// Draw the main UI
function drawUI() {
  g.clear();
  
  const GAP = 30;  // Gap between lines
  const bigtext = 3;
  const liltext = 2;
  let y = 10;
  
  
  // BIG TEXT //////////////////////////////////////////////////////////////////
  g.setFont("6x8", bigtext);
  
  // Corrected HR
  g.setColor(COLORS.corrected);
  g.drawString(`MLc ${correctedHR > 0 ? correctedHR : "--"}`, 10, y);
  y += GAP;
  
  // Raw HR
  g.setColor(COLORS.raw);
  g.drawString(`Raw ${rawHR > 0 ? rawHR : "--"}`, 10, y);
  y += 1.5*GAP;
  
  // lil text //////////////////////////////////////////////////////////////////
  g.setFont("6x8", liltext);
  
  // Buffer
  g.setColor(COLORS.raw);
  g.drawString("Buffer " + hrModel.hrBuffer.length + "/" + hrModel.bufferSize, 10, y);
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
    
    // Quick feedback
    Bangle.buzz(50);
  } else {
    console.log(`Logging stopped - ${logData.length} samples`);
    
    // Save automatically on stop
    if (logData.length > 0) {
      saveLog();
    }
    
    // Two buzzes for stop
    Bangle.buzz(50);
    setTimeout(() => Bangle.buzz(50), 150);
  }
  
  drawUI();
}

// Save log to file
function saveLog() {
  if (logData.length === 0) {
    return;
  }
  
  // Format as CSV
  let csv = "timestamp,raw_hr,confidence,mlc_hr\n";
  for (let entry of logData) {
    csv += `${entry.timestamp},${entry.raw},${entry.confidence},${entry.correction}\n`;
  }
  
  // Write to storage with timestamp
  const filename = `hrmodel_${Date.now()}.csv`;
  require("Storage").write(filename, csv);
  
  console.log(`Saved ${logData.length} samples to ${filename}`);
}

// Handle physical button press
Bangle.on('lcdPower', function(on) {
  if (on && !showingWelcome) {
    // Button pressed while screen is on - toggle logging
    toggleLogging();
  }
});

// Alternative: use setWatch for button
setWatch(function() {
  if (!showingWelcome) {
    toggleLogging();
  }
}, BTN1, {repeat: true, edge: "falling"});

// Handle touch events
Bangle.on('touch', function(button, xy) {
  if (showingWelcome) {
    // Any tap dismisses welcome
    showingWelcome = false;
    drawUI();
    return;
  }
});

// Handle HRM data
Bangle.on('HRM', function(hrm) {
  // Don't process while showing welcome
  if (showingWelcome) return;

  rawHR = hrm.bpm;
  confidence = hrm.confidence;
  if (rawHR > 0) {
    correctedHR = hrModel.getCorrectedHR(rawHR);
    if (isLogging) {
      logData.push({
        timestamp: Math.round(Date.now()),
        raw: rawHR,
        corrected: correctedHR,
        confidence: confidence
      });
    }
  }

  drawUI();
});

// Initial setup
g.clear();
Bangle.setHRMPower(1); // Turn on heart rate monitor
Bangle.setLCDPower(1); // Keep screen on

// Show welcome screen
drawWelcome();

// Cleanup on exit
E.on('kill', function() {
  Bangle.setHRMPower(0);
});