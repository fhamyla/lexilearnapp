import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, Alert, StyleSheet, ActivityIndicator } from 'react-native';
import { Shield, Lock, ArrowRight, RefreshCw, LogOut } from 'lucide-react-native';

interface TwoFactorProps {
  email: string;
  onVerify: () => void;
  onCancel: () => void;
}

export const TwoFactorAuth: React.FC<TwoFactorProps> = ({ email, onVerify, onCancel }) => {
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [timer, setTimer] = useState(30);

  useEffect(() => {
    // Simulate sending email on mount
    const timeout = setTimeout(() => {
        Alert.alert("Demo Code", `A verification code has been sent to ${email}.\n\nUse code: 123456`);
    }, 500);

    const interval = setInterval(() => {
      setTimer((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);

    return () => {
        clearTimeout(timeout);
        clearInterval(interval);
    };
  }, [email]);

  const handleSubmit = () => {
    setError('');
    setIsLoading(true);

    // Simulate network delay
    setTimeout(() => {
      if (code === '123456') {
        onVerify();
      } else {
        setError('Invalid verification code. Please try again.');
        setIsLoading(false);
      }
    }, 1000);
  };

  const handleResend = () => {
    setTimer(30);
    Alert.alert("Code Resent", `A new verification code has been sent to ${email}.\n\nUse code: 123456`);
  };

  return (
    <View style={styles.container}>
      <View style={styles.card}>
        <View style={styles.header}>
          <View style={styles.iconCircle}>
            <Shield size={40} {...({color: "#4A90E2"} as any)} />
          </View>
          <Text style={styles.title}>Two-Factor Authentication</Text>
          <Text style={styles.subtitle}>
            To secure your account, please enter the 6-digit code sent to <Text style={styles.emailText}>{email}</Text>
          </Text>
        </View>

        <View style={styles.form}>
          <View>
            <Text style={styles.label}>Verification Code</Text>
            <View style={styles.inputContainer}>
              <View style={styles.inputIcon}>
                <Lock size={20} {...({color: "#9CA3AF"} as any)} />
              </View>
              <TextInput 
                value={code}
                onChangeText={(t) => setCode(t.replace(/[^0-9]/g, '').slice(0, 6))}
                style={styles.input}
                placeholder="000000"
                keyboardType="number-pad"
                maxLength={6}
                autoFocus
              />
            </View>
            {error ? <Text style={styles.errorText}>{error}</Text> : null}
          </View>

          <TouchableOpacity 
            onPress={handleSubmit}
            disabled={isLoading || code.length < 6}
            style={[styles.verifyButton, (isLoading || code.length < 6) && styles.disabledButton]}
          >
            <Text style={styles.verifyButtonText}>{isLoading ? 'Verifying...' : 'Verify'}</Text>
            {!isLoading && <ArrowRight size={20} {...({color: "#FFF"} as any)} />}
          </TouchableOpacity>
        </View>

        <View style={styles.footer}>
          <TouchableOpacity 
            onPress={handleResend}
            disabled={timer > 0}
            style={styles.resendButton}
          >
            <RefreshCw size={16} {...({color: timer > 0 ? '#9CA3AF' : '#4A90E2'} as any)} />
            <Text style={[styles.resendText, timer > 0 ? styles.textGray : styles.textBlue]}>
                 {timer > 0 ? `Resend Code in ${timer}s` : 'Resend Code'}
            </Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            onPress={onCancel}
            style={styles.cancelButton}
          >
            <LogOut size={16} {...({color: "#6B7280"} as any)} />
            <Text style={styles.cancelText}>Cancel & Log Out</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FDFBF7',
    padding: 16,
  },
  card: {
    backgroundColor: '#FFFFFF',
    padding: 32,
    borderRadius: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 10,
    width: '100%',
    maxWidth: 400,
    borderWidth: 2,
    borderColor: 'rgba(74, 144, 226, 0.2)',
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  iconCircle: {
    width: 80,
    height: 80,
    backgroundColor: '#DBEAFE',
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1F2937',
    textAlign: 'center',
  },
  subtitle: {
    color: '#6B7280',
    textAlign: 'center',
    marginTop: 8,
  },
  emailText: {
    fontWeight: 'bold',
    color: '#374151',
  },
  form: {
    gap: 24,
  },
  label: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#374151',
    marginBottom: 8,
  },
  inputContainer: {
    position: 'relative',
    justifyContent: 'center',
  },
  inputIcon: {
    position: 'absolute',
    left: 16,
    zIndex: 10,
  },
  input: {
    width: '100%',
    paddingLeft: 48,
    paddingRight: 16,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#E5E7EB',
    textAlign: 'center',
    fontSize: 24,
    fontWeight: 'bold',
    letterSpacing: 4,
    backgroundColor: '#FFFFFF',
  },
  errorText: {
    color: '#EF4444',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
    marginTop: 8,
  },
  verifyButton: {
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
    gap: 8,
  },
  disabledButton: {
    opacity: 0.5,
  },
  verifyButtonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 18,
  },
  footer: {
    marginTop: 24,
    alignItems: 'center',
    gap: 16,
  },
  resendButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  resendText: {
    fontWeight: 'bold',
    fontSize: 14,
  },
  textGray: {
    color: '#9CA3AF',
  },
  textBlue: {
    color: '#4A90E2',
  },
  cancelButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  cancelText: {
    color: '#6B7280',
    fontWeight: '600',
    fontSize: 14,
  },
});