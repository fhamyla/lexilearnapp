
import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, ScrollView, ActivityIndicator, StyleSheet, StatusBar, Platform, Alert, Image, Dimensions, ImageBackground } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Screen, Difficulty, UserProfile, ProgressRecord } from './types';
import { TRACING_ITEMS, READING_ITEMS, SPELLING_ITEMS, MEMORY_ITEMS } from './constants';
import { TracingActivity } from './components/TracingActivity';
import { ReadingActivity } from './components/ReadingActivity';
import { SpellingActivity } from './components/SpellingActivity';
import { MemoryActivity } from './components/MemoryActivity';
import { AssessmentFlow } from './components/AssessmentFlow';
import { LoginScreen } from './components/LoginScreen';
import { SignUpScreen } from './components/SignUpScreen';
import { ParentDashboard } from './components/ParentDashboard';
import { LearningJourney } from './components/LearningJourney';
import { AdminDashboard } from './components/AdminDashboard';
import { Brain, User, Map as MapIcon, GraduationCap, ArrowLeft, LogOut, PenTool, BookOpen, Keyboard, Zap, ChevronRight, Star, AlertTriangle, RefreshCw, Trophy } from 'lucide-react-native';
import { auth, db, onAuthStateChanged, signOut, doc, updateDoc, onSnapshot } from './firebaseConfig';
import * as ScreenOrientation from 'expo-screen-orientation';


const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Local Assets
const readingImg = require('./assets/Reading.png');
const writingImg = require('./assets/Writing.png');
const spellingImg = require('./assets/Spelling.png');
const memoryImg = require('./assets/memorySpan.png');

const App: React.FC = () => {
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const loadingRef = useRef(true); 
  const [currentScreen, setCurrentScreen] = useState<Screen>(Screen.LOGIN);
  const currentScreenRef = useRef<Screen>(Screen.LOGIN);
  const previousStatusRef = useRef<string | null>(null);

  const [difficulty, setDifficulty] = useState<Difficulty>(Difficulty.MILD);
  const [connectionStatus, setConnectionStatus] = useState<'ONLINE' | 'OFFLINE_MODE'>('ONLINE');
  const [retryTrigger, setRetryTrigger] = useState(0);

  const setScreen = (screen: Screen) => {
      setCurrentScreen(screen);
      currentScreenRef.current = screen;
  };

  const updateLoading = (val: boolean) => {
      setLoading(val);
      loadingRef.current = val;
  };

  useEffect(() => {
  if (currentScreen === Screen.CHILD_DASHBOARD) {
    ScreenOrientation.lockAsync(
      ScreenOrientation.OrientationLock.LANDSCAPE
    );
  } else {
    ScreenOrientation.lockAsync(
      ScreenOrientation.OrientationLock.PORTRAIT
    );
  }

  return () => {
    ScreenOrientation.lockAsync(
      ScreenOrientation.OrientationLock.PORTRAIT
    );
  };
}, [currentScreen]);


  useEffect(() => {
    const checkBackend = async () => {
        try {
            const response = await fetch('https://lexilearnapp.onrender.com');
            if (response.ok) console.log("✅ Backend is UP");
        } catch (error) {
            console.error("❌ Backend error", error);
        }
    };
    checkBackend();
  }, []);

  const saveProfileLocally = async (profile: UserProfile) => {
      try {
          await AsyncStorage.setItem(`user_profile_${profile.uid}`, JSON.stringify(profile));
      } catch (e) {
          console.error("Failed to save profile locally", e);
      }
  };

  const handleSignOut = async () => {
      try {
        if (currentUser) {
            await AsyncStorage.removeItem(`user_profile_${currentUser.uid}`);
        }
        await signOut(auth);
      } catch (e) {
          console.error("Sign out error", e);
      }
      setCurrentUser(null);
      setScreen(Screen.LOGIN);
      previousStatusRef.current = null;
  };

  const navigateBasedOnUser = (user: UserProfile) => {
      if (user.role === 'Admin') {
          setScreen(Screen.ADMIN_DASHBOARD);
          return;
      }
      if (user.status === 'PENDING') {
          handleSignOut();
          return;
      }
      if (user.status === 'REJECTED') {
          Alert.alert("Access Denied", "Your account application was rejected.");
          handleSignOut();
          return;
      }
      if (!user.assessmentComplete) {
          setScreen(Screen.ASSESSMENT);
      } else {
          setScreen(Screen.HOME);
      }
  };

  const handleRetryConnection = () => {
      updateLoading(true);
      setConnectionStatus('ONLINE');
      setRetryTrigger(prev => prev + 1);
  };
  
  const handleLoginSuccess = async (manualUser?: UserProfile) => {
      await AsyncStorage.removeItem('needs_reauth');
      if (manualUser) {
          await saveProfileLocally(manualUser);
          setCurrentUser(manualUser);
          navigateBasedOnUser(manualUser);
      } else if (currentUser) {
          navigateBasedOnUser(currentUser);
      }
  };

  useEffect(() => {
    const unsubscribeAuth = onAuthStateChanged(auth, async (user: any) => {
      if (user) {
        setConnectionStatus('ONLINE');
        try {
            const localData = await AsyncStorage.getItem(`user_profile_${user.uid}`);
            if (localData) {
                const parsedUser = JSON.parse(localData) as UserProfile;
                if (parsedUser.status === 'PENDING' && currentScreenRef.current !== Screen.SIGN_UP) {
                    handleSignOut();
                    updateLoading(false);
                    return;
                }
                if (currentScreenRef.current !== Screen.SIGN_UP) {
                    setCurrentUser(parsedUser);
                    setDifficulty(parsedUser.assignedDifficulty || Difficulty.MILD);
                    previousStatusRef.current = parsedUser.status;
                    if (loadingRef.current || currentScreen === Screen.LOGIN) {
                        navigateBasedOnUser(parsedUser);
                    }
                }
            }
        } catch (e) {
            console.log("Error loading local data:", e);
        }

        const userDocRef = doc(db, "users", user.uid);
        const unsubscribeSnapshot = onSnapshot(userDocRef, async (docSnap: any) => {
          if (docSnap.exists()) {
              const userData = docSnap.data() as UserProfile;
              const currentStatus = userData.status;
              const prevStatus = previousStatusRef.current;
              if (currentScreenRef.current === Screen.ADMIN_DASHBOARD) return;
              if (prevStatus === 'PENDING' && currentStatus === 'APPROVED') {
                   await handleSignOut();
                   updateLoading(false);
                   return;
              }
              if (currentStatus === 'PENDING' && currentScreenRef.current !== Screen.SIGN_UP) {
                  await handleSignOut();
                  updateLoading(false);
                  return;
              }
              setCurrentUser(userData);
              setDifficulty(userData.assignedDifficulty || Difficulty.MILD);
              saveProfileLocally(userData);
              previousStatusRef.current = currentStatus;
              if (loadingRef.current) {
                  navigateBasedOnUser(userData);
              } else {
                 if (currentStatus === 'REJECTED') {
                     handleSignOut();
                     Alert.alert("Access Revoked", "Your account has been rejected.");
                 } else if (currentStatus === 'APPROVED' && !userData.assessmentComplete && currentScreenRef.current !== Screen.ASSESSMENT) {
                     setScreen(Screen.ASSESSMENT);
                 }
              }
              updateLoading(false);
          } else if (currentScreenRef.current !== Screen.SIGN_UP) {
             await handleSignOut();
             updateLoading(false);
          }
        }, (error: any) => {
           if (error.code === 'permission-denied') {
               setConnectionStatus('OFFLINE_MODE');
               updateLoading(false);
           }
        });
        return () => unsubscribeSnapshot();
      } else {
        if (currentUser && currentUser.uid.startsWith('admin_manual_')) {
            updateLoading(false);
        } else {
            setCurrentUser(null);
            setScreen(Screen.LOGIN);
            updateLoading(false);
            previousStatusRef.current = null;
        }
      }
    });
    return unsubscribeAuth;
  }, [retryTrigger]);

  const handleActivityComplete = async (type: 'Tracing' | 'Reading' | 'Spelling' | 'Memory', score: number) => {
      const newRecord: ProgressRecord = {
          date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          activityType: type,
          score: score,
          details: `${type} Activity`
      };
      if (currentUser) {
          const updatedHistory = [...(currentUser.progressHistory || []), newRecord];
          const updatedUser = { ...currentUser, progressHistory: updatedHistory };
          setCurrentUser(updatedUser);
          saveProfileLocally(updatedUser);
          if (connectionStatus === 'ONLINE' && !currentUser.uid.startsWith('admin_manual_')) {
            try { await updateDoc(doc(db, "users", currentUser.uid), { progressHistory: updatedHistory }); } catch (e) {}
          }
      }
  };

  const handleLevelProgress = async (type: 'Tracing' | 'Reading' | 'Spelling' | 'Memory', levelIndex: number) => {
      if (!currentUser) return;
      const currentSaved = (currentUser[`last${type}Index` as keyof UserProfile] as number) ?? -1;
      if (levelIndex <= currentSaved) return;
      const fieldName = `last${type}Index` as keyof UserProfile;
      const updatedUser = { ...currentUser, [fieldName]: levelIndex };
      setCurrentUser(updatedUser);
      saveProfileLocally(updatedUser);
      if (connectionStatus === 'ONLINE' && !currentUser.uid.startsWith('admin_manual_')) {
        try { await updateDoc(doc(db, "users", currentUser.uid), { [fieldName]: levelIndex }); } catch (e) {}
      }
  };

  const getInitialIndex = (lastIndex?: number) => (lastIndex ?? -1) + 1;

  const renderChildDashboard = () => (
    <View style={styles.landscapeWrapper}>
      {/* Top Header Row - Compact Landscape */}
      <View style={styles.landscapeHeader}>
        <TouchableOpacity onPress={() => setScreen(Screen.HOME)} style={styles.landscapeHeaderBtn}>
            <ArrowLeft {...({color: "#6B7280"} as any)} size={20} />
            <Text style={styles.landscapeHeaderBtnText}>Back</Text>
        </TouchableOpacity>
        
        <View style={styles.landscapeTitleRow}>
            <Brain size={24} {...({color: "#4A90E2"} as any)} />
            <Text style={styles.landscapeHeaderTitle}>Student Playground</Text>
        </View>

        <TouchableOpacity onPress={() => setScreen(Screen.LEARNING_JOURNEY)} style={styles.landscapeBadge}>
            <MapIcon size={14} {...({color: "#2563EB"} as any)} />
            <Text style={styles.landscapeBadgeText}>My Map</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.landscapeInfoBar}>
          <Text style={styles.landscapeWelcomeText}>Hi {currentUser?.childName}!</Text>
          <View style={styles.landscapeLevelIndicator}>
            <Star size={12} {...({color: "#FACC15"} as any)} {...({fill: "#FACC15"} as any)} />
            <Text style={styles.landscapeLevelText}>{difficulty}</Text>
          </View>
      </View>

      {/* Horizontal Carousel with even smaller posters (180px) and zero overlapping */}
      <ScrollView 
          horizontal 
          showsHorizontalScrollIndicator={false} 
          contentContainerStyle={styles.horizontalScrollContainer}
          style={styles.horizontalScrollStyle}
          decelerationRate="fast"
          snapToInterval={200} // Card width (180) + Margin (20)
          snapToAlignment="center"
      >
        {/* READING POSTER */}
        <TouchableOpacity onPress={() => setScreen(Screen.READING)} style={styles.activityPosterSmall}>
          <ImageBackground source={readingImg} style={styles.posterImage} imageStyle={{ borderRadius: 20 }}>
            <View style={styles.posterOverlayMinimal}>
                <View style={styles.posterTop}>
                    <View style={styles.posterIconBoxSmall}>
                        <BookOpen size={20} {...({color: "#374151"} as any)} />
                    </View>
                </View>
                <View style={styles.posterBottomTextCompact}>
                    <Text style={styles.posterTitleMicro}>READING</Text>
                    <View style={styles.posterLevelRowSmall}>
                        <Text style={styles.posterLevelLabelMicro}>Lv {getInitialIndex(currentUser?.lastReadingIndex) + 1}</Text>
                        <Trophy size={14} {...({color: "#FACC15"} as any)} />
                    </View>
                </View>
            </View>
          </ImageBackground>
        </TouchableOpacity>

        {/* WRITING POSTER */}
        <TouchableOpacity onPress={() => setScreen(Screen.TRACING)} style={styles.activityPosterSmall}>
          <ImageBackground source={writingImg} style={styles.posterImage} imageStyle={{ borderRadius: 20 }}>
            <View style={styles.posterOverlayMinimal}>
                <View style={styles.posterTop}>
                    <View style={styles.posterIconBoxSmall}>
                        <PenTool size={20} {...({color: "#374151"} as any)} />
                    </View>
                </View>
                <View style={styles.posterBottomTextCompact}>
                    <Text style={styles.posterTitleMicro}>WRITING</Text>
                    <View style={styles.posterLevelRowSmall}>
                        <Text style={styles.posterLevelLabelMicro}>Lv {getInitialIndex(currentUser?.lastTracingIndex) + 1}</Text>
                        <Trophy size={14} {...({color: "#FACC15"} as any)} />
                    </View>
                </View>
            </View>
          </ImageBackground>
        </TouchableOpacity>

        {/* SPELLING POSTER */}
        <TouchableOpacity onPress={() => setScreen(Screen.SPELLING)} style={styles.activityPosterSmall}>
          <ImageBackground source={spellingImg} style={styles.posterImage} imageStyle={{ borderRadius: 20 }}>
            <View style={styles.posterOverlayMinimal}>
                <View style={styles.posterTop}>
                    <View style={styles.posterIconBoxSmall}>
                        <Keyboard size={20} {...({color: "#374151"} as any)} />
                    </View>
                </View>
                <View style={styles.posterBottomTextCompact}>
                    <Text style={styles.posterTitleMicro}>SPELLING</Text>
                    <View style={styles.posterLevelRowSmall}>
                        <Text style={styles.posterLevelLabelMicro}>Lv {getInitialIndex(currentUser?.lastSpellingIndex) + 1}</Text>
                        <Trophy size={14} {...({color: "#FACC15"} as any)} />
                    </View>
                </View>
            </View>
          </ImageBackground>
        </TouchableOpacity>

        {/* MEMORY SPAN POSTER */}
        <TouchableOpacity onPress={() => setScreen(Screen.MEMORY)} style={styles.activityPosterSmall}>
          <ImageBackground source={memoryImg} style={styles.posterImage} imageStyle={{ borderRadius: 20 }}>
            <View style={styles.posterOverlayMinimal}>
                <View style={styles.posterTop}>
                    <View style={styles.posterIconBoxSmall}>
                        <Zap size={20} {...({color: "#374151"} as any)} />
                    </View>
                </View>
                <View style={styles.posterBottomTextCompact}>
                    <Text style={styles.posterTitleMicro}>MEMORY</Text>
                    <View style={styles.posterLevelRowSmall}>
                        <Text style={styles.posterLevelLabelMicro}>Lv {getInitialIndex(currentUser?.lastMemoryIndex) + 1}</Text>
                        <Trophy size={14} {...({color: "#FACC15"} as any)} />
                    </View>
                </View>
            </View>
          </ImageBackground>
        </TouchableOpacity>
      </ScrollView>

      <View style={styles.landscapeSwipeHint}>
          <Text style={styles.swipeHintText}>Swipe horizontally to see all adventures</Text>
          <ChevronRight size={12} {...({color: "#9CA3AF"} as any)} />
      </View>
    </View>
  );

  const renderLanding = () => (
    <View style={styles.landingContainer}>
      <TouchableOpacity onPress={handleSignOut} style={styles.signOutButton}>
          <LogOut size={20} {...({color: "#6B7280"} as any)} />
          <Text style={{color: '#6B7280', marginLeft: 8}}>Sign Out</Text>
      </TouchableOpacity>
      <View style={{flexDirection: 'row', alignItems: 'center', marginBottom: 64}}>
          <Brain size={64} {...({color: "#4A90E2"} as any)} />
          <Text style={styles.logoText}>LexiLearn</Text>
      </View>
      <View style={{width: '100%', gap: 32}}>
        <TouchableOpacity onPress={() => setScreen(Screen.CHILD_DASHBOARD)} style={styles.roleCard}>
          <View style={[styles.roleIconBox, {backgroundColor: '#DBEAFE'}]}><User size={48} {...({color: "#4A90E2"} as any)} /></View>
          <Text style={styles.roleTitle}>Student Area</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => setScreen(Screen.DASHBOARD)} style={[styles.roleCard, {borderColor: 'rgba(22, 163, 74, 0.2)'}]}>
          <View style={[styles.roleIconBox, {backgroundColor: '#DCFCE7'}]}><GraduationCap size={48} {...({color: "#16A34A"} as any)} /></View>
          <Text style={styles.roleTitle}>Parent Dashboard</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  if (loading) return <View style={styles.center}><ActivityIndicator size="large" color="#4A90E2" /></View>;

  return (
    <View style={styles.container}>
       <StatusBar barStyle="dark-content" />
       {connectionStatus === 'OFFLINE_MODE' && (
           <View style={styles.offlineBanner}>
               <View style={styles.bannerRow}><AlertTriangle size={16} {...({color: "#B45309"} as any)} /><Text style={styles.offlineText}>Offline Mode</Text></View>
               <TouchableOpacity onPress={handleRetryConnection} style={styles.retryButton}><RefreshCw size={14} {...({color: "#78350F"} as any)} /><Text style={styles.retryText}>Retry</Text></TouchableOpacity>
           </View>
       )}
       {currentScreen === Screen.LOGIN && <LoginScreen onSignUpClick={() => setScreen(Screen.SIGN_UP)} onLoginSuccess={handleLoginSuccess} />}
       {currentScreen === Screen.SIGN_UP && <SignUpScreen onBack={() => setScreen(Screen.LOGIN)} />}
       {currentScreen === Screen.ASSESSMENT && currentUser && <AssessmentFlow user={currentUser} onComplete={(p) => { setCurrentUser(p); setDifficulty(p.assignedDifficulty); saveProfileLocally(p); setScreen(Screen.HOME); }} />}
       {currentScreen === Screen.HOME && renderLanding()}
       {currentScreen === Screen.CHILD_DASHBOARD && renderChildDashboard()}
       {currentScreen === Screen.TRACING && <TracingActivity items={TRACING_ITEMS} difficulty={difficulty} initialIndex={getInitialIndex(currentUser?.lastTracingIndex)} onComplete={(s) => handleActivityComplete('Tracing', s)} onLevelComplete={(idx) => handleLevelProgress('Tracing', idx)} onExit={() => setScreen(Screen.CHILD_DASHBOARD)} />}
       {currentScreen === Screen.READING && <ReadingActivity items={READING_ITEMS} difficulty={difficulty} initialIndex={getInitialIndex(currentUser?.lastReadingIndex)} onComplete={(s) => handleActivityComplete('Reading', s)} onLevelComplete={(idx) => handleLevelProgress('Reading', idx)} onExit={() => setScreen(Screen.CHILD_DASHBOARD)} />}
       {currentScreen === Screen.SPELLING && <SpellingActivity items={SPELLING_ITEMS} difficulty={difficulty} initialIndex={getInitialIndex(currentUser?.lastSpellingIndex)} onComplete={(s) => handleActivityComplete('Spelling', s)} onLevelComplete={(idx) => handleLevelProgress('Spelling', idx)} onExit={() => setScreen(Screen.CHILD_DASHBOARD)} />}
       {currentScreen === Screen.MEMORY && <MemoryActivity items={MEMORY_ITEMS} difficulty={difficulty} initialIndex={getInitialIndex(currentUser?.lastMemoryIndex)} onComplete={(s) => handleActivityComplete('Memory', s)} onLevelComplete={(idx) => handleLevelProgress('Memory', idx)} onExit={() => setScreen(Screen.CHILD_DASHBOARD)} />}
       {currentScreen === Screen.DASHBOARD && currentUser && <ParentDashboard childName={currentUser.childName} currentDifficulty={difficulty} progressData={currentUser.progressHistory || []} assessmentScores={currentUser.assessmentScores} onUpdateChildName={(n) => {}} onExit={() => setScreen(Screen.HOME)} />}
       {currentScreen === Screen.LEARNING_JOURNEY && currentUser && <LearningJourney user={currentUser} onExit={() => setScreen(Screen.CHILD_DASHBOARD)} />}
       {currentScreen === Screen.ADMIN_DASHBOARD && currentUser && <AdminDashboard userEmail={currentUser.email} onExit={handleSignOut} />}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#FDFBF7', paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  landingContainer: { flex: 1, padding: 24, justifyContent: 'center', alignItems: 'center', backgroundColor: '#FDFBF7' },
  signOutButton: { position: 'absolute', top: 48, right: 24, flexDirection: 'row', alignItems: 'center' },
  logoText: { fontSize: 48, fontWeight: 'bold', color: '#2D2D2D', marginLeft: 12 },
  roleCard: { backgroundColor: '#FFFFFF', padding: 32, borderWidth: 2, borderColor: 'rgba(74, 144, 226, 0.2)', borderRadius: 24, shadowColor: '#000', shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.1, shadowRadius: 20, elevation: 5, alignItems: 'center' },
  roleIconBox: { padding: 24, borderRadius: 999, marginBottom: 24 },
  roleTitle: { fontSize: 24, fontWeight: 'bold', color: '#1F2937' },
  offlineBanner: { backgroundColor: '#FEF3C7', padding: 12, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', borderBottomWidth: 1, borderBottomColor: '#F59E0B' },
  bannerRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  offlineText: { color: '#B45309', fontWeight: 'bold', fontSize: 12 },
  retryButton: { flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: 'rgba(255,255,255,0.5)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
  retryText: { fontSize: 12, fontWeight: 'bold', color: '#78350F' },

  // --- COMPACT LANDSCAPE PLAYGROUND ---
  landscapeWrapper: { flex: 1, backgroundColor: '#FDFBF7', paddingVertical: 8 },
  landscapeHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 32, marginBottom: 6 },
  landscapeHeaderBtn: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  landscapeHeaderBtnText: { color: '#6B7280', fontWeight: 'bold', fontSize: 14 },
  landscapeTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  landscapeHeaderTitle: { fontSize: 20, fontWeight: '900', color: '#1F2937' },
  landscapeBadge: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#DBEAFE', paddingHorizontal: 10, paddingVertical: 5, borderRadius: 20, borderWidth: 1, borderColor: '#BFDBFE' },
  landscapeBadgeText: { marginLeft: 4, fontWeight: 'bold', color: '#1D4ED8', fontSize: 11 },
  landscapeInfoBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 32, marginBottom: 8 },
  landscapeWelcomeText: { fontSize: 14, color: '#4B5563' },
  landscapeLevelIndicator: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#FEF9C3', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 10, borderWidth: 1, borderColor: '#FDE68A' },
  landscapeLevelText: { color: '#854D0E', fontWeight: 'bold', marginLeft: 3, fontSize: 11 },
  horizontalScrollStyle: { flex: 1 },
  horizontalScrollContainer: { paddingHorizontal: 32, paddingVertical: 6, alignItems: 'center' },
  activityPosterSmall: { 
    width: 180, 
    height: 180, 
    marginRight: 20, 
    borderRadius: 20, 
    overflow: 'hidden', 
    backgroundColor: '#FFF', 
    shadowColor: '#000', 
    shadowOffset: { width: 0, height: 4 }, 
    shadowOpacity: 0.1, 
    shadowRadius: 6, 
    elevation: 4, 
    borderWidth: 1.5, 
    borderColor: '#FFF' 
  },
  posterImage: { width: '100%', height: '100%' },
  posterOverlayMinimal: { 
    flex: 1, 
    padding: 12, 
    justifyContent: 'space-between', 
    borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.01)' 
  },
  posterTop: { flexDirection: 'row', alignItems: 'center' },
  posterIconBoxSmall: { 
    width: 36, 
    height: 36, 
    borderRadius: 18, 
    backgroundColor: 'rgba(255,255,255,0.95)', 
    alignItems: 'center', 
    justifyContent: 'center', 
    shadowColor: '#000', 
    shadowOpacity: 0.05, 
    shadowRadius: 2 
  },
  posterBottomTextCompact: { 
    gap: 0,
    backgroundColor: 'rgba(0,0,0,0.3)', 
    padding: 8,
    borderRadius: 14,
    marginHorizontal: -4,
    marginBottom: -4
  },
  posterTitleMicro: { 
    color: '#FFF', 
    fontSize: 18, 
    fontWeight: '900', 
    textShadowColor: 'rgba(0, 0, 0, 0.4)', 
    textShadowOffset: { width: 1, height: 1 }, 
    textShadowRadius: 3 
  },
  posterLevelRowSmall: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  posterLevelLabelMicro: { 
    color: '#FFF', 
    fontSize: 11, 
    fontWeight: 'bold', 
    textShadowColor: 'rgba(0, 0, 0, 0.4)', 
    textShadowRadius: 2 
  },
  landscapeSwipeHint: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 4, marginTop: 4 },
  swipeHintText: { color: '#9CA3AF', fontSize: 10, fontWeight: '600' }
});

export default App;
