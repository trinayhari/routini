// Script to verify environment variables are accessible
console.log('Environment Check:');
console.log(`NEXT_PUBLIC_BACKEND_URL: ${process.env.NEXT_PUBLIC_BACKEND_URL || 'Not set (will default to http://localhost:8000)'}`); 