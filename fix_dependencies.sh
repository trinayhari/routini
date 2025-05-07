#!/bin/bash

# Install dependencies that might be missing
echo "Installing potential missing dependencies..."

# Frontend dependencies
npm install --save next-cors
npm install --save cross-fetch

# Print setup instructions
echo ""
echo "==== SETUP INSTRUCTIONS ===="
echo "1. Make sure the FastAPI backend is running:"
echo "   cd backend && python run.py"
echo ""
echo "2. In a new terminal, start the Next.js frontend:"
echo "   npm run dev"
echo ""
echo "3. Open your browser to: http://localhost:3000/compare"
echo ""
echo "4. If you see CORS errors in the console, install a CORS browser extension"
echo "   such as 'CORS Unblock' for Chrome or Firefox"
echo "============================" 