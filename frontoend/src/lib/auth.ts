import {
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signOut as firebaseSignOut,
    onAuthStateChanged,
    User as FirebaseUser,
    sendPasswordResetEmail,
    updateProfile,
  } from 'firebase/auth'
  import { auth } from './firebase'
  import { useEffect, useState } from 'react'

  // ユーザー型定義
  export interface User {
    uid: string
    email: string | null
    name: string | null
    photoURL?: string | null
  }

  // カスタムフック用の戻り値の型定義
  interface UseAuthReturn {
    user: User | null
    loading: boolean
    error: Error | null
    signIn: (email: string, password: string) => Promise<void>
    signUp: (email: string, password: string, name: string) => Promise<void>
    signOut: () => Promise<void>
    resetPassword: (email: string) => Promise<void>
    updateUserProfile: (displayName?: string, photoURL?: string) => Promise<void>
  }

  // Firebase ユーザーを独自のUser型に変換する関数
  const formatUser = (user: FirebaseUser): User => ({
    uid: user.uid,
    email: user.email,
    name: user.displayName,
    photoURL: user.photoURL,
  })

  // カスタム認証フック
  export const useAuth = (): UseAuthReturn => {
    const [user, setUser] = useState<User | null>(null)
    const [loading, setLoading] = useState<boolean>(true)
    const [error, setError] = useState<Error | null>(null)

    // 認証状態の監視
    useEffect(() => {
      const unsubscribe = onAuthStateChanged(
        auth,
        (user) => {
          setLoading(true)
          if (user) {
            setUser(formatUser(user))
          } else {
            setUser(null)
          }
          setError(null)
          setLoading(false)
        },
        (error) => {
          console.error('Auth state change error:', error)
          setError(error as Error)
          setLoading(false)
        }
      )

      return () => unsubscribe()
    }, [])

    // サインイン関数
    const signIn = async (email: string, password: string) => {
      try {
        setLoading(true)
        const userCredential = await signInWithEmailAndPassword(auth, email, password)
        setUser(formatUser(userCredential.user))
        setError(null)
      } catch (err) {
        setError(err as Error)
        throw err
      } finally {
        setLoading(false)
      }
    }

    // サインアップ関数
    const signUp = async (email: string, password: string, name: string) => {
      try {
        setLoading(true)
        const userCredential = await createUserWithEmailAndPassword(auth, email, password)

        // ユーザープロファイルの更新
        await updateProfile(userCredential.user, {
          displayName: name
        })

        setUser(formatUser(userCredential.user))
        setError(null)
      } catch (err) {
        setError(err as Error)
        throw err
      } finally {
        setLoading(false)
      }
    }

    // サインアウト関数
    const signOut = async () => {
      try {
        await firebaseSignOut(auth)
        setUser(null)
        setError(null)
      } catch (err) {
        setError(err as Error)
        throw err
      }
    }

    // パスワードリセット関数
    const resetPassword = async (email: string) => {
      try {
        await sendPasswordResetEmail(auth, email)
        setError(null)
      } catch (err) {
        setError(err as Error)
        throw err
      }
    }

    // ユーザープロファイル更新関数
    const updateUserProfile = async (displayName?: string, photoURL?: string) => {
      if (!auth.currentUser) {
        throw new Error('No user logged in')
      }

      try {
        await updateProfile(auth.currentUser, {
          displayName: displayName || auth.currentUser.displayName,
          photoURL: photoURL || auth.currentUser.photoURL
        })

        if (auth.currentUser) {
          setUser(formatUser(auth.currentUser))
        }

        setError(null)
      } catch (err) {
        setError(err as Error)
        throw err
      }
    }

    return {
      user,
      loading,
      error,
      signIn,
      signUp,
      signOut,
      resetPassword,
      updateUserProfile
    }
  }