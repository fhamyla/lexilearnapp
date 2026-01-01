
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as firebaseApp from "firebase/app";
import * as firebaseAuth from "firebase/auth";
import * as firebaseFirestore from "firebase/firestore";

// Workaround for TypeScript errors: cast modules to any to access functions
const appModule = firebaseApp as any;
const authModule = firebaseAuth as any;
const firestoreModule = firebaseFirestore as any;

// App functions
const initializeApp = appModule.initializeApp;
const getApp = appModule.getApp;
const getApps = appModule.getApps;

// Auth functions - Exporting them to use in other files to avoid import errors
export const getAuth = authModule.getAuth;
export const signInWithEmailAndPassword = authModule.signInWithEmailAndPassword;
export const createUserWithEmailAndPassword = authModule.createUserWithEmailAndPassword;
export const signOut = authModule.signOut;
export const onAuthStateChanged = authModule.onAuthStateChanged;
export const sendEmailVerification = authModule.sendEmailVerification;
export const sendPasswordResetEmail = authModule.sendPasswordResetEmail;
// Export initializeAuth and persistence helper for internal use or if needed elsewhere
export const initializeAuth = authModule.initializeAuth;
export const getReactNativePersistence = authModule.getReactNativePersistence;

// Firestore functions - Exporting them to resolve module errors
export const getFirestore = firestoreModule.getFirestore;
export const doc = firestoreModule.doc;
export const getDoc = firestoreModule.getDoc;
export const getDocs = firestoreModule.getDocs;
export const setDoc = firestoreModule.setDoc;
export const updateDoc = firestoreModule.updateDoc;
export const deleteDoc = firestoreModule.deleteDoc;
export const onSnapshot = firestoreModule.onSnapshot;
export const collection = firestoreModule.collection;
export const query = firestoreModule.query;
export const where = firestoreModule.where;
export const writeBatch = firestoreModule.writeBatch;
export const serverTimestamp = firestoreModule.serverTimestamp;
export const serverTimestampFn = firestoreModule.serverTimestamp;
// Export common Firestore types/functions if needed, but handled as any above so straightforward exports suffice.

// Firebase Configuration
const firebaseConfig = {
  apiKey: "AIzaSyDQkb_C0nfBV40amjICzQuBqtUr4qNWbpc",
  authDomain: "multisensorylearning-b553b.firebaseapp.com",
  projectId: "multisensorylearning-b553b",
  storageBucket: "multisensorylearning-b553b.firebasestorage.app",
  messagingSenderId: "1034000458938",
  appId: "1:1034000458938:web:eb202a95faddaac0e6ce83",
  measurementId: "G-R7B9Z3KH8J"
};

let app;
let auth: any;
let db: any; // Using any for db to avoid Firestore type conflicts in this workaround

try {
  // Prevent hot-reload errors in Expo by checking if app is already initialized
  if (getApps && getApps().length === 0) {
    app = initializeApp(firebaseConfig);
    
    // Initialize Auth with AsyncStorage persistence to prevent memory-only sessions
    if (initializeAuth && getReactNativePersistence) {
      try {
        auth = initializeAuth(app, {
          persistence: getReactNativePersistence(AsyncStorage)
        });
      } catch (e) {
        console.warn("Failed to initialize auth with persistence, falling back to default.", e);
        auth = getAuth(app);
      }
    } else {
      auth = getAuth(app);
    }

  } else {
    app = getApp();
    auth = getAuth(app);
  }
  
  db = getFirestore(app);

} catch (error) {
  console.error("Firebase initialization error:", error);
  throw error;
}

export { auth, db };
// Base URL for callable cloud functions that perform admin operations (generate verification links).
// Set this to your deployed Cloud Function or server endpoint.
export const FUNCTIONS_BASE_URL = '';
// Helper to encode emails for use as document IDs
export const encodeEmail = (email: string) => {
  return String(email || '').toLowerCase().replace(/\./g, ',');
};
