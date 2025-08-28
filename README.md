# Photo Sorter Hybrid

Hybrid face-sorting app using local photo loading and cloud-based embedding comparison.

## Architecture
- Local app handles: photo loading, local embedding, final saving
- Server (Oracle Cloud) handles: embedding comparison & sorting logic
- UI: web interface with placeholder for local download

## Structure
- `backend/`: Flask server API hosted on Oracle Cloud
- `local-app/`: Local CLI app for processing and interaction
- `ui/`: Static frontend served locally or via backend
