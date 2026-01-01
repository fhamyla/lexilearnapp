
import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, Modal, Pressable, ActivityIndicator, Alert, StyleSheet } from 'react-native';
import { Mail, User, Baby, ArrowLeft, Send, ChevronDown, Lock } from 'lucide-react-native';
import { auth, db, createUserWithEmailAndPassword, signOut, collection, doc, writeBatch } from '../firebaseConfig';
import { GuardianApplication, UserProfile, Difficulty } from '../types';

interface SignUpProps {
  onBack: () => void;
}

export const SignUpScreen: React.FC<SignUpProps> = ({ onBack }) => {
  const [formData, setFormData] = useState({
    guardianName: '',
    email: '',
    password: '',
    childName: '',
    childAge: '',
    relationship: 'Parent'
  });
  
  const [ratings, setRatings] = useState({
    reading: 0,
    writing: 0,
    spelling: 0
  });

  const [loading, setLoading] = useState(false);
  const [relationshipModalVisible, setRelationshipModalVisible] = useState(false);

  const handleSubmit = async () => {
    // 1. Validation
    if (!formData.guardianName || !formData.email || !formData.password || !formData.childName || !formData.childAge) {
      Alert.alert("Missing Information", "Please fill in all text fields.");
      return;
    }

    if (ratings.reading === 0 || ratings.writing === 0 || ratings.spelling === 0) {
        Alert.alert("Incomplete Assessment", "Please assign a priority (1st, 2nd, 3rd) to each skill.");
        return;
    }

    const priorities = [ratings.reading, ratings.writing, ratings.spelling];
    const unique = new Set(priorities);
    if (unique.size !== 3) {
       Alert.alert("Invalid Priorities", "Each skill must have a unique priority (1, 2, or 3).");
       return;
    }

    setLoading(true);

    try {
      // 2. Create Authentication User
      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      const uid = userCredential.user.uid;

      // 3. Prepare Data for Firestore
      const userProfile: UserProfile = {
          uid,
          email: formData.email,
          childName: formData.childName,
          childAge: formData.childAge,
          role: 'Guardian',
          status: 'PENDING',
          assessmentComplete: false,
          assignedDifficulty: Difficulty.MILD,
          progressHistory: []
      };

      // 4. Batch Write
      const batch = writeBatch(db);
      const userRef = doc(db, "users", uid);
      const appRef = doc(collection(db, "applications"));

      const application: GuardianApplication = {
          id: appRef.id, // Explicitly adding ID
          uid,
          guardianName: formData.guardianName,
          email: formData.email,
          childName: formData.childName,
          childAge: formData.childAge,
          relationship: formData.relationship,
          difficultyRatings: ratings,
          status: 'PENDING',
          dateApplied: new Date().toISOString()
      };

      batch.set(userRef, userProfile);
      batch.set(appRef, application);

      await batch.commit();

      // 5. Success Flow - Sign out and return to Login immediately
      await signOut(auth);
      onBack();

    } catch (error: any) {
        console.error(error);
        let msg = "Could not create account.";
        if (error.code === 'auth/email-already-in-use') msg = "Email is already in use.";
        if (error.code === 'auth/weak-password') msg = "Password should be at least 6 characters.";
        Alert.alert("Error", msg);
    } finally {
        setLoading(false);
    }
  };

  const handleRating = (skill: 'reading' | 'writing' | 'spelling', val: number) => {
    setRatings(prev => ({ ...prev, [skill]: val }));
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
           <ArrowLeft size={24} {...({color: "#4B5563"} as any)} />
           <Text style={styles.backText}>Back to Login</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Guardian Application</Text>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        
        {/* Guardian Info */}
        <View style={styles.section}>
            <Text style={styles.sectionTitle}>Guardian Information</Text>
            
            <View style={styles.inputGroup}>
                <Text style={styles.label}>Full Name</Text>
                <View style={styles.inputContainer}>
                    <View style={styles.inputIcon}>
                        <User size={20} {...({color: "#9CA3AF"} as any)} />
                    </View>
                    <TextInput 
                        style={styles.input}
                        value={formData.guardianName}
                        onChangeText={t => setFormData({...formData, guardianName: t})}
                        placeholder="John Doe"
                    />
                </View>
            </View>

            <View style={styles.inputGroup}>
                <Text style={styles.label}>Email Address</Text>
                <View style={styles.inputContainer}>
                    <View style={styles.inputIcon}>
                        <Mail size={20} {...({color: "#9CA3AF"} as any)} />
                    </View>
                    <TextInput 
                        style={styles.input}
                        value={formData.email}
                        onChangeText={t => setFormData({...formData, email: t})}
                        placeholder="john@example.com"
                        keyboardType="email-address"
                        autoCapitalize="none"
                    />
                </View>
            </View>

            <View style={styles.inputGroup}>
                <Text style={styles.label}>Password</Text>
                <View style={styles.inputContainer}>
                    <View style={styles.inputIcon}>
                        <Lock size={20} {...({color: "#9CA3AF"} as any)} />
                    </View>
                    <TextInput 
                        style={styles.input}
                        value={formData.password}
                        onChangeText={t => setFormData({...formData, password: t})}
                        placeholder="Min. 6 characters"
                        secureTextEntry
                    />
                </View>
            </View>

            <View style={styles.inputGroup}>
                <Text style={styles.label}>Relationship to Child</Text>
                <TouchableOpacity onPress={() => setRelationshipModalVisible(true)} style={styles.inputContainer}>
                    <View style={styles.inputIcon}>
                        <User size={20} {...({color: "#9CA3AF"} as any)} />
                    </View>
                    <Text style={styles.inputText}>{formData.relationship}</Text>
                    <ChevronDown size={20} {...({color: "#9CA3AF"} as any)} />
                </TouchableOpacity>
            </View>
        </View>

        {/* Modal for Relationship */}
        <Modal visible={relationshipModalVisible} transparent animationType="fade">
            <Pressable onPress={() => setRelationshipModalVisible(false)} style={styles.modalOverlay}>
                <View style={styles.modalContent}>
                    {['Parent', 'Guardian', 'Teacher', 'Therapist', 'Other'].map(r => (
                        <TouchableOpacity 
                           key={r} 
                           style={styles.modalItem}
                           onPress={() => {
                               setFormData({...formData, relationship: r});
                               setRelationshipModalVisible(false);
                           }}
                        >
                            <Text style={styles.modalText}>{r}</Text>
                        </TouchableOpacity>
                    ))}
                </View>
            </Pressable>
        </Modal>

        {/* Child Info */}
        <View style={styles.section}>
            <Text style={styles.sectionTitle}>Child Information</Text>
            
            <View style={styles.row}>
                <View style={[styles.inputGroup, {flex: 2, marginRight: 12}]}>
                    <Text style={styles.label}>Child Name</Text>
                    <View style={styles.inputContainer}>
                        <View style={styles.inputIcon}>
                            <Baby size={20} {...({color: "#9CA3AF"} as any)} />
                        </View>
                        <TextInput 
                            style={styles.input}
                            value={formData.childName}
                            onChangeText={t => setFormData({...formData, childName: t})}
                            placeholder="Name"
                        />
                    </View>
                </View>
                <View style={[styles.inputGroup, {flex: 1}]}>
                    <Text style={styles.label}>Age</Text>
                    <View style={styles.inputContainer}>
                        <TextInput 
                            style={[styles.input, {paddingLeft: 16, textAlign: 'center'}]}
                            value={formData.childAge}
                            onChangeText={t => setFormData({...formData, childAge: t})}
                            placeholder="Age"
                            keyboardType="numeric"
                        />
                    </View>
                </View>
            </View>
        </View>

        {/* Assessment */}
        <View style={styles.section}>
            <Text style={styles.sectionTitle}>Initial Skill Priorities</Text>
            <Text style={styles.sectionDesc}>Rank the focus areas (1 = Highest Priority)</Text>
            
            <View style={styles.ratingContainer}>
                {[
                    { key: 'reading', label: 'Reading' },
                    { key: 'writing', label: 'Writing' },
                    { key: 'spelling', label: 'Spelling' }
                ].map((item) => (
                    <View key={item.key} style={styles.ratingRow}>
                        <Text style={styles.ratingLabel}>{item.label}</Text>
                        <View style={styles.ratingButtons}>
                            {[1, 2, 3].map(num => (
                                <TouchableOpacity 
                                    key={num}
                                    onPress={() => handleRating(item.key as any, num)}
                                    style={[
                                        styles.ratingBtn,
                                        ratings[item.key as keyof typeof ratings] === num && styles.ratingBtnActive
                                    ]}
                                >
                                    <Text style={[
                                        styles.ratingBtnText,
                                        ratings[item.key as keyof typeof ratings] === num && styles.ratingBtnTextActive
                                    ]}>{num}</Text>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>
                ))}
            </View>
        </View>

        <TouchableOpacity onPress={handleSubmit} style={styles.submitButton} disabled={loading}>
            {loading ? <ActivityIndicator color="#FFF" /> : (
                <>
                    <Text style={styles.submitButtonText}>Submit Application</Text>
                    <Send size={20} {...({color: "#FFF"} as any)} />
                </>
            )}
        </TouchableOpacity>

      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FDFBF7',
  },
  header: {
    padding: 24,
    paddingTop: 60,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 8,
  },
  backText: {
    color: '#4B5563',
    fontWeight: 'bold',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  scrollContent: {
    padding: 24,
    paddingBottom: 48,
  },
  section: {
    marginBottom: 32,
    backgroundColor: '#FFFFFF',
    padding: 24,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 16,
  },
  sectionDesc: {
    color: '#6B7280',
    marginBottom: 16,
    fontSize: 14,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#374151',
    marginBottom: 8,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#D1D5DB',
    borderRadius: 12,
    backgroundColor: '#F9FAFB',
    paddingHorizontal: 12,
  },
  inputIcon: {
    marginRight: 8,
  },
  input: {
    flex: 1,
    paddingVertical: 12,
    fontSize: 16,
    color: '#1F2937',
  },
  inputText: {
    flex: 1,
    paddingVertical: 12,
    fontSize: 16,
    color: '#1F2937',
  },
  row: {
    flexDirection: 'row',
  },
  ratingContainer: {
    gap: 16,
  },
  ratingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  ratingLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#4B5563',
  },
  ratingButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  ratingBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#D1D5DB',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FFFFFF',
  },
  ratingBtnActive: {
    backgroundColor: '#4A90E2',
    borderColor: '#4A90E2',
  },
  ratingBtnText: {
    fontWeight: 'bold',
    color: '#6B7280',
  },
  ratingBtnTextActive: {
    color: '#FFFFFF',
  },
  submitButton: {
    backgroundColor: '#4A90E2',
    paddingVertical: 16,
    borderRadius: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 4,
  },
  submitButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  modalContent: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    width: '100%',
    maxWidth: 320,
    overflow: 'hidden',
  },
  modalItem: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
    alignItems: 'center',
  },
  modalText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
  },
});
