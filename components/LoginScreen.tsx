
import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ActivityIndicator, ScrollView, Modal, Pressable, Alert, StyleSheet } from 'react-native';
import { auth, db, signInWithEmailAndPassword, signOut, doc, getDoc, collection, query, where, getDocs, sendEmailVerification, sendPasswordResetEmail, setDoc, deleteDoc } from '../firebaseConfig';
import { FUNCTIONS_BASE_URL, encodeEmail, serverTimestamp } from '../firebaseConfig';
import { Brain, Lock, Mail, AlertCircle, Users, ClipboardList, ChevronDown, ShieldCheck } from 'lucide-react-native';
import { UserProfile, Difficulty } from '../types';

interface LoginProps {
  onSignUpClick?: () => void;
  onLoginSuccess: (user?: UserProfile) => void;
}

export const LoginScreen: React.FC<LoginProps> = ({ onSignUpClick, onLoginSuccess }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [roleModalVisible, setRoleModalVisible] = useState(false); // retained for compatibility but hidden

  const handleLogin = async () => {
    if (!email || !password) {
      setError("Please enter both email/username and password.");
      return;
    }

    setLoading(true);
    setError('');

    let loginIdentifier = email.trim();
    const cleanedPassword = password.trim();

    // 0. Username Resolution: if input does not contain '@', try to find email by username
    if (!loginIdentifier.includes('@')) {
        try {
            const usersRef = collection(db, "users");
            const q = query(usersRef, where("username", "==", loginIdentifier));
            const snapshot = await getDocs(q);
            
            if (snapshot.empty) {
                setError("Username not found.");
                setLoading(false);
                return;
            }
            
            const userData = snapshot.docs[0].data() as UserProfile;
            if (userData.email) {
                loginIdentifier = userData.email;
            }
        } catch (err: any) {
            console.error("Username lookup failed", err);
            // If offline or permission issue, we can't look up username.
            // But we will let it fail at authentication step or show specific error.
            if (err.code === 'permission-denied') {
                setError("Connection error. Please check internet.");
                setLoading(false);
                return;
            }
        }
    }

    // Attempt standard Firebase Auth sign-in first
    try {
      const userCredential = await signInWithEmailAndPassword(auth, loginIdentifier, cleanedPassword);
      const uid = userCredential.user.uid;

      // Determine if signed-in user is an admin — admins are allowed without email verification
      let isAdmin = false;
      try {
        const userDocRef = doc(db, 'users', uid);
        const userSnapCheck = await getDoc(userDocRef);
        if (userSnapCheck.exists()) {
          const ud = userSnapCheck.data() as any;
          if (ud.role && (ud.role === 'Admin' || ud.role === 'Administrator')) isAdmin = true;
        }
      } catch (e) {
        // ignore lookup errors
      }

      if (!isAdmin) {
        try {
          const adminsRef = collection(db, 'admin');
          const aQ = query(adminsRef, where('email', '==', loginIdentifier));
          const aSnap = await getDocs(aQ);
          if (!aSnap.empty) isAdmin = true;
        } catch (e) {
          // ignore
        }
      }

      // If not admin, enforce email verification
      if (!isAdmin && auth.currentUser && auth.currentUser.emailVerified === false) {
        try { await sendEmailVerification(auth.currentUser); } catch (e) { /* ignore */ }
        await signOut(auth);
        Alert.alert('Email Not Verified', 'Please verify your email address. A verification email was (re)sent.');
        setLoading(false);
        return;
      }

      // Fetch user profile from Firestore
      const userDocRef = doc(db, 'users', uid);
      const userSnap = await getDoc(userDocRef);

      if (userSnap.exists()) {
        const userData = userSnap.data() as UserProfile;
        // If account is not APPROVED, block
        if (userData.status === 'PENDING') {
          await signOut(auth);
          Alert.alert('Application Pending', 'Your account is waiting for administrator approval.');
          setLoading(false);
          return;
        }
        if (userData.status === 'REJECTED') {
          await signOut(auth);
          Alert.alert('Access Denied', 'Your application was rejected.');
          setLoading(false);
          return;
        }

        // Successful login
        onLoginSuccess(userData);
        return;
      }

      // If no user doc, check pending_verifications (user verified recently) before admin list
      try {
        const pvRef = doc(db, 'pending_verifications', uid);
        const pvSnap = await getDoc(pvRef);
        if (pvSnap.exists()) {
          const pv = pvSnap.data() as any;
          const sentAt = pv.sentAt;
          const now = Date.now();
          const sentMs = sentAt && sentAt.toMillis ? sentAt.toMillis() : (new Date(pv.sentAt).getTime ? new Date(pv.sentAt).getTime() : 0);
          const elapsed = now - sentMs;
          const TWO_MIN = 2 * 60 * 1000;
          if (elapsed <= TWO_MIN && auth.currentUser && auth.currentUser.emailVerified) {
            // Verification within window — finalize creation
            const userProfile: UserProfile = {
              uid,
              email: pv.email,
              childName: pv.childName,
              childAge: pv.childAge,
              role: 'Guardian',
              status: 'PENDING',
              assessmentComplete: false,
              assignedDifficulty: Difficulty.MILD,
              progressHistory: []
            };
            const appRef = doc(collection(db, 'applications'));
            const application: any = {
              id: appRef.id,
              uid,
              guardianName: pv.guardianName,
              email: pv.email,
              childName: pv.childName,
              childAge: pv.childAge,
              relationship: pv.relationship,
              difficultyRatings: pv.difficultyRatings,
              status: 'PENDING',
              dateApplied: new Date().toISOString()
            };
            try {
              await setDoc(doc(db, 'users', uid), userProfile);
              await setDoc(appRef, application);
            } catch (e) {
              console.warn('Failed to finalize pending verification', e);
            }
            try { await deleteDoc(pvRef); } catch (e) { /* ignore */ }
            onLoginSuccess(userProfile);
            return;
          } else {
            // Expired — remove account as failure to verify
            try { await auth.currentUser?.delete(); } catch (e) { /* ignore */ }
            try { await deleteDoc(pvRef); } catch (e) { /* ignore */ }
            await signOut(auth);
            setError('Verification expired. Account removed. Please sign up again.');
            setLoading(false);
            return;
          }
        }
      } catch (e) {
        console.warn('pending_verifications check failed', e);
      }

      // If no pending verification, check admin manual list
      try {
        const adminsRef = collection(db, 'admin');
        const q = query(adminsRef, where('email', '==', loginIdentifier));
        const snap = await getDocs(q);
        if (!snap.empty) {
          // Allow admin access (user authenticated via Firebase Auth but missing users doc)
          const adminProfile: UserProfile = {
            uid,
            email: loginIdentifier,
            childName: 'Administrator',
            role: 'Admin',
            status: 'APPROVED',
            assessmentComplete: true,
            assignedDifficulty: Difficulty.MILD
          };
          onLoginSuccess(adminProfile);
          return;
        }
      } catch (e) {
        // ignore permission errors here
      }

      // If reached here, account data not found
      await signOut(auth);
      setError('Account data not found.');
      setLoading(false);
      return;

    } catch (err: any) {
      // If Auth failed, attempt manual admin check (admins collection with password)
      try {
        const adminsRef = collection(db, 'admin');
        const q = query(adminsRef, where('email', '==', loginIdentifier), where('password', '==', cleanedPassword));
        const snap = await getDocs(q);
        if (!snap.empty) {
          const adminProfile: UserProfile = {
            uid: 'admin_manual_' + Date.now(),
            email: loginIdentifier,
            childName: 'Administrator',
            role: 'Admin',
            status: 'APPROVED',
            assessmentComplete: true,
            assignedDifficulty: Difficulty.MILD
          };
          onLoginSuccess(adminProfile);
          return;
        }
      } catch (e) {
        // ignore
      }

      // Generic error message for users (hide system internals)
      setError('Invalid credentials.');
      setLoading(false);
      return;
    }
  };

  // Forgot password handler
  const handleForgotPassword = async () => {
    if (!email) {
      Alert.alert('Missing Email', 'Please enter your email or username first.');
      return;
    }

    let targetEmail = email.trim();
    if (!targetEmail.includes('@')) {
      // resolve username
      try {
        const usersRef = collection(db, 'users');
        const q = query(usersRef, where('username', '==', targetEmail));
        const snap = await getDocs(q);
        if (snap.empty) {
          Alert.alert('Not Found', 'Username not found.');
          return;
        }
        targetEmail = snap.docs[0].data().email;
      } catch (e) {
        Alert.alert('Error', 'Could not look up username.');
        return;
      }
    }

    // Prefer server-side endpoint to enforce admin-block and cooldown
      if (FUNCTIONS_BASE_URL) {
      try {
        const resp = await fetch(`${FUNCTIONS_BASE_URL}/sendPasswordReset`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: targetEmail })
        });
        const data = await resp.json();
        if (resp.ok && data.sent) {
          Alert.alert('Reset Email Sent', 'A password reset link has been sent if the email exists.');
          return;
        } else if (data.error === 'admin') {
          Alert.alert('Not Allowed', 'Password reset for admin accounts is not allowed through this form.');
          return;
        } else if (data.error === 'cooldown') {
          Alert.alert('Try Later', 'Password reset was recently requested. Please wait 2 minutes before trying again.');
          return;
        } else {
          Alert.alert('Error', 'Could not send reset email.');
          return;
        }
      } catch (e) {
        console.warn('Password reset endpoint failed, falling back to client reset', e);
      }
    }

    // Fallback: client-side reset (can't enforce admin/cooldown reliably)
      try {
        // cooldown check (best-effort client-side)
        try {
          const metaRef = doc(db, 'password_reset_meta', encodeEmail(targetEmail));
          const metaSnap = await getDoc(metaRef);
          if (metaSnap.exists()) {
            const last = metaSnap.data()?.lastResetAt;
            if (last && (Date.now() - last.toDate().getTime()) < 2 * 60 * 1000) {
              Alert.alert('Try Later', 'Password reset was recently requested. Please wait 2 minutes before trying again.');
              return;
            }
          }
          await sendPasswordResetEmail(auth, targetEmail);
          await setDoc(doc(db, 'password_reset_meta', encodeEmail(targetEmail)), { lastResetAt: serverTimestamp() });
          Alert.alert('Reset Email Sent', 'A password reset link has been sent if the email exists.');
        } catch (e) {
          console.warn('Password reset fallback failed', e);
          Alert.alert('Error', 'Could not send reset email.');
        }
      } catch (e) {
        Alert.alert('Error', 'Could not send reset email.');
      }
  };

  return (
    <ScrollView contentContainerStyle={styles.scrollContainer} style={styles.container}>
      <View style={styles.card}>
        <View style={styles.header}>
          <View style={styles.logoContainer}>
            <Brain size={48} {...({color: "#4A90E2"} as any)} />
          </View>
          <Text style={styles.title}>LexiLearn</Text>
          <Text style={styles.subtitle}>Dyslexia Support Platform</Text>
        </View>

        <View style={styles.form}>
          {error ? (
            <View style={styles.errorBox}>
              <AlertCircle size={20} {...({color: "#DC2626"} as any)} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          {/* Role selection removed — system auto-detects account type at login */}

          <View>
            <Text style={styles.label}>Email or Username</Text>
            <View style={styles.inputWrapper}>
              <View style={styles.iconPos}>
                <Mail size={20} {...({color: "#9CA3AF"} as any)} />
              </View>
              <TextInput 
                value={email}
                onChangeText={setEmail}
                style={styles.textInput}
                placeholder="Email or Username"
                placeholderTextColor="#9CA3AF"
                autoCapitalize="none"
              />
            </View>
          </View>

          <View>
            <Text style={styles.label}>Password</Text>
            <View style={styles.inputWrapper}>
               <View style={styles.iconPos}>
                 <Lock size={20} {...({color: "#9CA3AF"} as any)} />
               </View>
              <TextInput 
                value={password}
                onChangeText={setPassword}
                style={styles.textInput}
                placeholder="••••••••"
                placeholderTextColor="#9CA3AF"
                secureTextEntry
              />
            </View>
          </View>

          <TouchableOpacity onPress={handleForgotPassword} style={{ alignSelf: 'flex-end', marginTop: 8 }}>
            <Text style={{ color: '#2563EB' }}>Forgot password?</Text>
          </TouchableOpacity>

          <TouchableOpacity 
            onPress={handleLogin} 
            disabled={loading}
            style={styles.loginButton}
          >
            {loading ? <ActivityIndicator color="#FFF" /> : <Text style={styles.loginButtonText}>Log In</Text>}
          </TouchableOpacity>
        </View>

        <View style={styles.signUpContainer}>
           <TouchableOpacity 
             onPress={onSignUpClick}
             style={styles.signUpButton}
           >
              <ClipboardList size={20} {...({color: "#7E22CE"} as any)} /> 
              <Text style={styles.signUpText}>Don't have an account? Apply Here</Text>
           </TouchableOpacity>
        </View>

        <Text style={styles.footerText}>
          Guardians must be approved by an Admin.
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FDFBF7',
  },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 16,
  },
  card: {
    backgroundColor: '#FFFFFF',
    padding: 24,
    borderRadius: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 5,
    width: '100%',
    borderWidth: 2,
    borderColor: 'rgba(74, 144, 226, 0.2)',
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  logoContainer: {
    padding: 16,
    backgroundColor: '#DBEAFE',
    borderRadius: 999,
    marginBottom: 16,
  },
  title: {
    fontSize: 30,
    fontWeight: 'bold',
    color: '#2D2D2D',
  },
  subtitle: {
    color: '#6B7280',
    marginTop: 4,
  },
  form: {
    gap: 16,
  },
  errorBox: {
    padding: 16,
    backgroundColor: '#FEF2F2',
    borderColor: '#FECACA',
    borderWidth: 1,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  errorText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#DC2626',
    flex: 1,
  },
  label: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#374151',
    marginBottom: 8,
  },
  inputContainer: {
    position: 'relative',
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    paddingLeft: 48,
    paddingRight: 16,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#E5E7EB',
    backgroundColor: '#FFFFFF',
  },
  iconPos: {
    position: 'absolute',
    left: 16,
    zIndex: 10,
  },
  inputText: {
    flex: 1,
    color: '#1F2937',
    fontSize: 16,
  },
  modalOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 16,
  },
  modalContent: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    width: '100%',
    maxWidth: 384,
    overflow: 'hidden',
  },
  modalHeader: {
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 16,
    color: '#9CA3AF',
    paddingVertical: 12,
    backgroundColor: '#F9FAFB',
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  modalItem: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
    flexDirection: 'row',
    alignItems: 'center',
  },
  modalText: {
    fontWeight: 'bold',
    fontSize: 18,
    color: '#374151',
  },
  inputWrapper: {
    position: 'relative',
    justifyContent: 'center',
  },
  textInput: {
    width: '100%',
    paddingLeft: 48,
    paddingRight: 16,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#E5E7EB',
    color: '#1F2937',
    fontSize: 16,
    backgroundColor: '#FFFFFF',
  },
  loginButton: {
    width: '100%',
    paddingVertical: 16,
    backgroundColor: '#4A90E2',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 8,
  },
  loginButtonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 18,
  },
  signUpContainer: {
    marginTop: 24,
  },
  signUpButton: {
    width: '100%',
    paddingVertical: 12,
    backgroundColor: '#F3E8FF',
    borderWidth: 2,
    borderColor: '#F3E8FF',
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  signUpText: {
    color: '#7E22CE',
    fontWeight: 'bold',
  },
  footerText: {
    marginTop: 16,
    textAlign: 'center',
    fontSize: 14,
    color: '#9CA3AF',
  },
});
