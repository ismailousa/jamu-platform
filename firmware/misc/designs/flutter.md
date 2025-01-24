# AI-Assisted Prescreening Tool

## 1. Welcome & Login/Registration

**Purpose:** First-time setup and user authentication.

**Key Features:**
- Simple login or one-time registration.
- Quick access for existing users.

**Design:**
```css
[ App Logo ]
[ Welcome Message: "Welcome to the AI-Assisted Prescreening Tool" ]
[ Buttons: "Login" | "Register" ]
```

## 2. Wireless Camera Setup

**Purpose:** Ensure quick and easy camera pairing.

**Workflow:**
- Detect nearby wireless cameras automatically.
- Tap to pair and test the connection.
- Confirm successful pairing with live preview.

**Key Features:**
- Step-by-step pairing guide.
- Visual connection indicator.

**Design:**
```css
[ Header: "Set Up Your Camera" ]
[ Status: "Searching for cameras..." (with animation) ]
[ Detected Cameras List: "Camera 1", "Camera 2" ]
[ Buttons: "Pair" | "Retry" ]
[ Live Feed Preview After Successful Connection ]
```

## 3. Home Screen

**Purpose:** Central hub for the prescreening process.

**Key Features:**
- Start screening immediately or access previous sessions.
- Button for patient management.

**Design:**
```css
[ Header: "Welcome, Dr. [Name]" ]
[ Buttons: "Start Screening" | "View Records" | "Settings" ]
```

## 4. Screening Workflow

### A. Start Screening (Live Feed Screen)

**Purpose:** Capture diagnostic images with real-time AI preprocessing.

**Workflow:**
- Select body part to screen (e.g., eyes, throat).
- Adjust camera feed using brightness/contrast sliders if needed.
- Capture images as guided by AI overlays.

**Key Features:**
- Real-time feedback (e.g., alignment hints).
- One-tap image capture.

**Design:**
```css
[ Top Bar: Patient Name, Body Part Selected ]
[ Live Feed with AI Overlay (e.g., alignment guide) ]
[ Brightness/Contrast Sliders (if needed) ]
[ Capture Button (large, centered at bottom) ]
```

### B. Patient Information Entry

**Purpose:** Collect or confirm patient data.

**Workflow:**
- Autofill details for existing patients.
- Add new details (name, age, gender) if needed.

**Key Features:**
- Auto-save session details.

**Design:**
```css
[ Header: "Patient Information" ]
[ Fields: - Name - Age - Gender ]
[ Buttons: "Save & Continue" | "Edit Details" ]
```

### C. Screening Results

**Purpose:** View processed images and save results.

**Workflow:**
- Display captured image with AI analysis overlay.
- Allow editing or adding notes.

**Key Features:**
- Save or retake image options.

**Design:**
```css
[ Captured Image with Analysis Overlay ]
[ Notes Section (Optional) ]
[ Buttons: "Save Image" | "Retake" ]
```

## 5. Records/Report Management

**Purpose:** View, edit, and share previous screening records.

**Key Features:**
- Search and filter by patient name or date.
- Quick export options for images and reports.

**Design:**
```css
[ Header: "Records" ]
[ Search Bar ]
[ List of Records (Name, Date, Body Part) ]
[ Buttons: "View Details" | "Export" ]
```

## 6. Settings

**Purpose:** Customize the app and manage preferences.

**Key Features:**
- Camera preferences: resolution, filters.
- Account management: logout, change password.

**Design:**
```css
[ Header: "Settings" ]
[ Sections: - Camera Settings - Account Settings - Theme & Language Preferences ]
```

## Simplified Workflow for Doctors

- **Login/Register:** Authenticate and access the home screen.
- **Camera Setup:** Pair the camera in just a few taps.
- **Start Screening:** Capture diagnostic images with AI guidance.
- **Save Results:** Save results alongside patient details.
- **View/Export Records:** Manage saved data and generate reports.
- **Adjust Settings:** Personalize app behavior and preferences.

## Key Improvements for Usability

- **Intuitive Camera Setup:** Automatic detection and guided pairing reduce frustration.
- **Minimal Input for Screening:** Doctors spend less time filling out details and more on capturing data.
- **AI Assistance:** Real-time overlays guide doctors to capture accurate diagnostic images.
- **Streamlined Records:** Easy access to previous sessions ensures continuity of care.