
import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView, ActivityIndicator, Alert, StyleSheet, Linking } from 'react-native';
import { GuardianApplication } from '../types';
import { db, collection, query, getDocs, doc, updateDoc, FUNCTIONS_BASE_URL } from '../firebaseConfig';
import { Shield, CheckCircle, XCircle, Clock, Search, LogOut, Mail, User, Baby, BookOpen, PenTool, Type, AlertCircle, RefreshCw } from 'lucide-react-native';

interface AdminDashboardProps {
  userEmail: string;
  onExit: () => void;
}

export const AdminDashboard: React.FC<AdminDashboardProps> = ({ userEmail, onExit }) => {
  const [applications, setApplications] = useState<GuardianApplication[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'ALL' | 'PENDING' | 'APPROVED'>('PENDING');

  useEffect(() => {
    fetchApplications();
  }, []);

  const fetchApplications = async () => {
    setLoading(true);
    setError(null);
    try {
      const q = query(collection(db, 'applications'));
      const snapshot = await getDocs(q);
      const apps = snapshot.docs.map((doc: any) => ({ id: doc.id, ...doc.data() } as GuardianApplication));
      
      // Client-side sort
      apps.sort((a, b) => new Date(b.dateApplied).getTime() - new Date(a.dateApplied).getTime());
      
      setApplications(apps);
    } catch (error: any) {
      console.error("Error fetching applications:", error);
      if (error.code === 'permission-denied') {
          setError("Access Denied: You do not have permissions to view applications.");
      } else {
          setError("Could not load applications. Please check your connection.");
      }
    } finally {
      setLoading(false);
    }
  };

  const generateUsername = (fullName: string) => {
    const cleanName = fullName.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    const randomSuffix = Math.floor(1000 + Math.random() * 9000);
    return `${cleanName}${randomSuffix}`;
  };

  const handleStatusChange = async (app: GuardianApplication, newStatus: 'APPROVED' | 'REJECTED') => {
    if (!app.uid || !app.id) {
        Alert.alert("Error", "User ID missing for this application. Cannot update user record.");
        return;
    }

    const processUpdate = async () => {
        try {
            let generatedUsername = "";
            
            // Generate Username if approving and one doesn't exist
            if (newStatus === 'APPROVED') {
                generatedUsername = app.username || generateUsername(app.guardianName);
            }

            // Step 2: Update Application Status and Username
            const appRef = doc(db, 'applications', app.id!);
            const updateData: any = { status: newStatus };
            if (generatedUsername) updateData.username = generatedUsername;
            
            await updateDoc(appRef, updateData);
      
            // Update User Profile Status in 'users' collection
            const userRef = doc(db, 'users', app.uid!);
            await updateDoc(userRef, updateData);
      
            setApplications(prev => prev.map(a => 
              a.id === app.id ? { ...a, status: newStatus, username: generatedUsername } : a
            ));
            
            if (newStatus === 'APPROVED') {
              // Call server/cloud function to send verification and schedule auto-deletion if unverified
              const sendVerificationAndSchedule = async () => {
                try {
                  if (!FUNCTIONS_BASE_URL) {
                    // No server endpoint configured — fallback to manual mailto
                    const subject = 'LexiLearn Account Approved - Login Details';
                    const body = `Dear ${app.guardianName},\n\nCongratulations! Your Guardian Application for LexiLearn has been approved.\n\nPlease verify your email address by following the link that will be sent to you.\n\nUsername: ${generatedUsername}\n\nBest,\nLexiLearn Admin Team`;
                    const mailUrl = `mailto:${app.email}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
                    Linking.openURL(mailUrl).catch(() => {});
                    onExit();
                    return;
                  }

                  const resp = await fetch(`${FUNCTIONS_BASE_URL}/approveGuardian`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      uid: app.uid,
                      email: app.email,
                      username: generatedUsername,
                      applicationId: app.id
                    })
                  });

                  if (!resp.ok) {
                    console.warn('approveGuardian returned non-ok', resp.status);
                    Alert.alert('Approved', 'Account approved. Verification could not be sent automatically.');
                  } else {
                    const data = await resp.json();
                    if (data.sent) {
                      Alert.alert('Verification Sent', 'A verification email was sent to the user. They must verify within 2 minutes.');
                    } else if (data.link) {
                      // open link or notify
                      Linking.openURL(data.link).catch(() => {});
                      Alert.alert('Verification Link', 'Verification link opened.');
                    } else {
                      Alert.alert('Approved', 'Account approved. Verification action returned unknown status.');
                    }
                  }

                } catch (err) {
                  console.error('Error sending verification:', err);
                  Alert.alert('Approved', 'Account approved. Verification could not be sent automatically.');
                } finally {
                  onExit();
                }
              };

              // Auto-send verification when admin clicks Approve
              sendVerificationAndSchedule();
            } else {
              Alert.alert('Rejected', 'Application has been rejected.', [{ text: 'OK', onPress: () => onExit() }]);
            }
      
          } catch (error: any) {
            console.error("Error updating status:", error);
            if (error.code === 'permission-denied') {
                Alert.alert("Permission Error", "You do not have permission to perform this action.");
            } else {
                Alert.alert("Error", "Failed to update status. Please try again.");
            }
          }
    };

    if (newStatus === 'APPROVED') {
        Alert.alert(
            "Confirm Approval",
            `Are you sure you want to approve access for ${app.guardianName}?`,
            [
                { text: "Cancel", style: "cancel" },
                { text: "Approve Access", onPress: processUpdate }
            ]
        );
    } else {
        Alert.alert(
            "Confirm Rejection",
            `Are you sure you want to REJECT ${app.guardianName}? They will not be able to log in.`,
            [
                { text: "Cancel", style: "cancel" },
                { text: "Reject", style: "destructive", onPress: processUpdate }
            ]
        );
    }
  };

  const filteredApps = applications.filter(app => {
    if (filter === 'ALL') return true;
    return app.status === filter;
  });

  const getRatingStyle = (rating: number) => {
      if (rating === 1) return { bg: '#FEE2E2', text: '#B91C1C' };
      if (rating === 2) return { bg: '#FEF3C7', text: '#B45309' };
      return { bg: '#DCFCE7', text: '#15803D' };
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.shieldIcon}>
            <Shield {...({color: "#FFF"} as any)} size={24} />
          </View>
          <View>
            <Text style={styles.headerTitle}>Admin Dashboard</Text>
            <Text style={styles.headerSubtitle}>Logged in as {userEmail}</Text>
          </View>
        </View>
        <View style={styles.headerRight}>
             <TouchableOpacity 
              onPress={fetchApplications}
              style={styles.refreshButton}
            >
              <RefreshCw size={20} {...({color: "#374151"} as any)} />
            </TouchableOpacity>
            <TouchableOpacity 
              onPress={onExit}
              style={styles.logoutButton}
            >
              <LogOut size={20} {...({color: "#374151"} as any)} />
            </TouchableOpacity>
        </View>
      </View>

      {error ? (
          <View style={styles.errorBar}>
              <AlertCircle size={20} {...({color: "#FFF"} as any)} />
              <Text style={styles.errorBarText}>{error} - Check Firestore Rules</Text>
          </View>
      ) : null}

      <ScrollView contentContainerStyle={{ paddingBottom: 24 }} showsVerticalScrollIndicator={false}>
            {/* Stats Row */}
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.statsScroll} contentContainerStyle={styles.statsContainer}>
            <View style={styles.statCard}>
                <View style={styles.statRow}>
                <View>
                    <Text style={styles.statLabel}>Pending</Text>
                    <Text style={[styles.statValue, {color: '#2563EB'}]}>{applications.filter(a => a.status === 'PENDING').length}</Text>
                </View>
                <View style={[styles.statIcon, {backgroundColor: '#EFF6FF'}]}>
                    <Clock size={24} {...({color: "#3B82F6"} as any)} />
                </View>
                </View>
            </View>
            
            <View style={styles.statCard}>
                <View style={styles.statRow}>
                <View>
                    <Text style={styles.statLabel}>Total Users</Text>
                    <Text style={[styles.statValue, {color: '#16A34A'}]}>{applications.filter(a => a.status === 'APPROVED').length}</Text>
                </View>
                <View style={[styles.statIcon, {backgroundColor: '#ECFDF5'}]}>
                    <User size={24} {...({color: "#10B981"} as any)} />
                </View>
                </View>
            </View>
            
            <View style={styles.statCard}>
                <View style={styles.statRow}>
                <View>
                    <Text style={styles.statLabel}>System Status</Text>
                    <Text style={styles.statusText}>Active</Text>
                </View>
                <View style={[styles.statIcon, {backgroundColor: '#F5F3FF'}]}>
                    <Shield size={24} {...({color: "#8B5CF6"} as any)} />
                </View>
                </View>
            </View>
            </ScrollView>

            {/* Filters */}
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterScroll}>
            {['PENDING', 'APPROVED', 'ALL'].map((f) => (
                <TouchableOpacity
                key={f}
                onPress={() => setFilter(f as any)}
                style={[
                    styles.filterButton,
                    filter === f ? styles.filterActive : styles.filterInactive
                ]}
                >
                <Text style={[styles.filterText, filter === f ? styles.filterTextActive : styles.filterTextInactive]}>
                    {f === 'ALL' ? 'All Applications' : f.charAt(0) + f.slice(1).toLowerCase()}
                </Text>
                </TouchableOpacity>
            ))}
            </ScrollView>

            {/* List */}
            <View style={styles.listContainer}>
            {loading ? (
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color="#4A90E2" />
                    <Text style={styles.loadingText}>Loading applications from database...</Text>
                </View>
            ) : filteredApps.length === 0 ? (
                <View style={styles.loadingContainer}>
                <Search size={48} {...({color: "#E5E7EB"} as any)} />
                <Text style={styles.emptyText}>No applications found matching this filter.</Text>
                </View>
            ) : (
                <View>
                    {filteredApps.map((app, index) => (
                    <View key={app.id || index} style={styles.applicationItem}>
                        
                        {/* Header Line */}
                        <View style={styles.appHeader}>
                            <View>
                                <Text style={styles.guardianName}>{app.guardianName}</Text>
                                {app.status === 'PENDING' && (
                                  <View style={styles.emailRow}>
                                    <Mail size={14} {...({color: "#6B7280"} as any)} />
                                    <Text style={styles.emailText}>{app.email}</Text>
                                  </View>
                                )}
                                {app.username && (
                                    <View style={styles.emailRow}>
                                        <User size={14} {...({color: "#6B7280"} as any)} />
                                        <Text style={styles.emailText}>Username: {app.username}</Text>
                                    </View>
                                )}
                            </View>
                            <View style={[
                                styles.statusBadge,
                                app.status === 'PENDING' ? {backgroundColor: '#DBEAFE'} :
                                app.status === 'APPROVED' ? {backgroundColor: '#DCFCE7'} : {backgroundColor: '#FEE2E2'}
                            ]}>
                                <Text style={[
                                    styles.statusTextBadge,
                                    app.status === 'PENDING' ? {color: '#1D4ED8'} :
                                    app.status === 'APPROVED' ? {color: '#15803D'} : {color: '#B91C1C'}
                                ]}>{app.status}</Text>
                            </View>
                        </View>

                        {/* Child Info */}
                        <View style={styles.childInfoSection}>
                            <View style={styles.infoRow}>
                                <Baby size={16} {...({color: "#6B7280"} as any)} />
                                <Text style={styles.infoLabel}>Child:</Text>
                                <Text style={styles.infoValue}>{app.childName} (Age: {app.childAge})</Text>
                            </View>
                            <View style={styles.infoRow}>
                                <User size={16} {...({color: "#6B7280"} as any)} />
                                <Text style={styles.infoLabel}>Relationship:</Text>
                                <Text style={styles.infoValue}>{app.relationship}</Text>
                            </View>
                        </View>

                        {/* Ratings */}
                        <View style={styles.ratingsContainer}>
                            <Text style={styles.ratingsTitle}>Priorities (1=High, 3=Low)</Text>
                            <View style={styles.ratingsRow}>
                                <View style={[styles.ratingBadge, { backgroundColor: getRatingStyle(app.difficultyRatings.reading).bg }]}>
                                    <BookOpen size={14} {...({color: getRatingStyle(app.difficultyRatings.reading).text} as any)} />
                                    <Text style={[styles.ratingText, { color: getRatingStyle(app.difficultyRatings.reading).text }]}>Reading: {app.difficultyRatings.reading}</Text>
                                </View>
                                <View style={[styles.ratingBadge, { backgroundColor: getRatingStyle(app.difficultyRatings.writing).bg }]}>
                                    <PenTool size={14} {...({color: getRatingStyle(app.difficultyRatings.writing).text} as any)} />
                                    <Text style={[styles.ratingText, { color: getRatingStyle(app.difficultyRatings.writing).text }]}>Writing: {app.difficultyRatings.writing}</Text>
                                </View>
                                <View style={[styles.ratingBadge, { backgroundColor: getRatingStyle(app.difficultyRatings.spelling).bg }]}>
                                    <Type size={14} {...({color: getRatingStyle(app.difficultyRatings.spelling).text} as any)} />
                                    <Text style={[styles.ratingText, { color: getRatingStyle(app.difficultyRatings.spelling).text }]}>Spelling: {app.difficultyRatings.spelling}</Text>
                                </View>
                            </View>
                        </View>

                        {/* Actions */}
                        {app.status === 'PENDING' && (
                            <View style={styles.actionsRow}>
                                <TouchableOpacity 
                                    onPress={() => handleStatusChange(app, 'REJECTED')}
                                    style={styles.rejectButton}
                                >
                                    <XCircle size={18} {...({color: "#DC2626"} as any)} />
                                    <Text style={styles.rejectText}>Reject</Text>
                                </TouchableOpacity>
                                <TouchableOpacity 
                                    onPress={() => handleStatusChange(app, 'APPROVED')}
                                    style={styles.approveButton}
                                >
                                    <CheckCircle size={18} {...({color: "#FFF"} as any)} />
                                    <Text style={styles.approveText}>Approve Access</Text>
                                </TouchableOpacity>
                            </View>
                        )}
                        {app.status === 'APPROVED' && (
                             <View style={styles.approvedInfo}>
                                <CheckCircle size={16} {...({color: "#16A34A"} as any)} />
                                <Text style={styles.approvedText}>Access Granted</Text>
                             </View>
                        )}
                         {app.status === 'REJECTED' && (
                             <View style={styles.rejectedInfo}>
                                <XCircle size={16} {...({color: "#DC2626"} as any)} />
                                <Text style={styles.rejectedText}>Application Rejected</Text>
                             </View>
                        )}
                    </View>
                    ))}
                </View>
            )}
            </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F3F4F6',
  },
  header: {
    backgroundColor: '#FFFFFF',
    padding: 24,
    paddingTop: 60,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  headerRight: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 12
  },
  shieldIcon: {
    backgroundColor: '#4A90E2',
    padding: 8,
    borderRadius: 12,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#6B7280',
  },
  refreshButton: {
    padding: 8,
    backgroundColor: '#F3F4F6',
    borderRadius: 8,
  },
  logoutButton: {
    padding: 8,
    backgroundColor: '#F3F4F6',
    borderRadius: 8,
  },
  errorBar: {
    backgroundColor: '#EF4444',
    padding: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  errorBarText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 14,
  },
  statsScroll: {
    paddingVertical: 16,
  },
  statsContainer: {
    paddingHorizontal: 16,
    gap: 12,
  },
  statCard: {
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 16,
    width: 160,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  statLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6B7280',
    marginBottom: 4,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  statusText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#8B5CF6',
  },
  statIcon: {
    padding: 8,
    borderRadius: 8,
  },
  filterScroll: {
    paddingHorizontal: 16,
    marginBottom: 16,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    borderWidth: 1,
  },
  filterActive: {
    backgroundColor: '#4A90E2',
    borderColor: '#4A90E2',
  },
  filterInactive: {
    backgroundColor: '#FFFFFF',
    borderColor: '#D1D5DB',
  },
  filterText: {
    fontSize: 14,
    fontWeight: '600',
  },
  filterTextActive: {
    color: '#FFFFFF',
  },
  filterTextInactive: {
    color: '#6B7280',
  },
  listContainer: {
    paddingHorizontal: 16,
  },
  loadingContainer: {
    padding: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    marginTop: 16,
    color: '#6B7280',
  },
  emptyText: {
    marginTop: 16,
    color: '#9CA3AF',
    fontSize: 16,
    textAlign: 'center',
  },
  applicationItem: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  appHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  guardianName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  emailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 4,
  },
  emailText: {
    color: '#6B7280',
    fontSize: 14,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  statusTextBadge: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  childInfoSection: {
    backgroundColor: '#F9FAFB',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    gap: 8,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  infoLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4B5563',
    width: 100,
  },
  infoValue: {
    fontSize: 14,
    color: '#1F2937',
    flex: 1,
  },
  ratingsContainer: {
    marginBottom: 16,
  },
  ratingsTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#9CA3AF',
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  ratingsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  ratingBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
  },
  ratingText: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  actionsRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  rejectButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#FEE2E2',
  },
  rejectText: {
    color: '#DC2626',
    fontWeight: 'bold',
  },
  approveButton: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#2563EB',
  },
  approveText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  approvedInfo: {
      marginTop: 8,
      padding: 12,
      backgroundColor: '#F0FDF4',
      borderRadius: 8,
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
      justifyContent: 'center'
  },
  approvedText: {
      color: '#16A34A',
      fontWeight: 'bold'
  },
  rejectedInfo: {
      marginTop: 8,
      padding: 12,
      backgroundColor: '#FEF2F2',
      borderRadius: 8,
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
      justifyContent: 'center'
  },
  rejectedText: {
      color: '#DC2626',
      fontWeight: 'bold'
  }
});
