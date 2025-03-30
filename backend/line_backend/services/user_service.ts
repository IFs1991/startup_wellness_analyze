import { hash, compare } from 'bcrypt';
import { sign, verify } from 'jsonwebtoken';
import { User, UserRole } from '../models/user_model';
import logger from '../../utils/logger';

const SALT_ROUNDS = 10;
const JWT_SECRET = process.env.JWT_SECRET || 'default-secret-change-in-production';
const TOKEN_EXPIRY = '24h';

interface UserCreationParams {
  lineUserId?: string;
  companyId: string;
  name: string;
  email: string;
  password: string;
  role?: UserRole;
}

interface UserUpdateParams {
  name?: string;
  email?: string;
  password?: string;
  role?: UserRole;
}

class UserService {
  /**
   * ユーザーを登録する
   */
  async registerUser(params: UserCreationParams): Promise<User> {
    try {
      logger.info(`Creating new user with email: ${params.email}`);
      
      // メールアドレスの重複チェック
      const existingUser = await User.findOne({ where: { email: params.email } });
      if (existingUser) {
        logger.warn(`User registration failed: Email ${params.email} already exists`);
        throw new Error('Email already in use');
      }
      
      // パスワードのハッシュ化
      const hashedPassword = await hash(params.password, SALT_ROUNDS);
      
      // ユーザーの作成
      const user = await User.create({
        lineUserId: params.lineUserId,
        companyId: params.companyId,
        name: params.name,
        email: params.email,
        password: hashedPassword,
        role: params.role || UserRole.USER
      });
      
      logger.info(`User created successfully with ID: ${user.id}`);
      return user;
    } catch (error) {
      logger.error(`Error creating user: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }

  /**
   * ユーザー認証を行う
   */
  async authenticateUser(email: string, password: string): Promise<{ user: User; token: string }> {
    try {
      logger.info(`Authenticating user with email: ${email}`);
      
      // ユーザーの検索
      const user = await User.findOne({ where: { email } });
      if (!user) {
        logger.warn(`Authentication failed: User with email ${email} not found`);
        throw new Error('Invalid email or password');
      }
      
      // パスワードの検証
      const isPasswordValid = await compare(password, user.password);
      if (!isPasswordValid) {
        logger.warn(`Authentication failed: Invalid password for user ${email}`);
        throw new Error('Invalid email or password');
      }
      
      // JWTトークンの生成
      const token = sign(
        { 
          userId: user.id, 
          email: user.email, 
          role: user.role 
        },
        JWT_SECRET,
        { expiresIn: TOKEN_EXPIRY }
      );
      
      logger.info(`User ${email} authenticated successfully`);
      return { user, token };
    } catch (error) {
      logger.error(`Authentication error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }

  /**
   * ユーザー情報を取得する
   */
  async getUserById(id: string): Promise<User> {
    try {
      logger.info(`Fetching user with ID: ${id}`);
      
      const user = await User.findByPk(id);
      if (!user) {
        logger.warn(`User not found with ID: ${id}`);
        throw new Error('User not found');
      }
      
      return user;
    } catch (error) {
      logger.error(`Error fetching user: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }

  /**
   * LINEユーザーIDでユーザーを取得する
   */
  async getUserByLineId(lineUserId: string): Promise<User | null> {
    try {
      logger.info(`Fetching user with LINE user ID: ${lineUserId}`);
      
      const user = await User.findOne({ where: { lineUserId } });
      return user;
    } catch (error) {
      logger.error(`Error fetching user by LINE ID: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }

  /**
   * 企業IDに基づいてユーザーを取得する
   */
  async getUsersByCompany(companyId: string): Promise<User[]> {
    try {
      logger.info(`Fetching users for company ID: ${companyId}`);
      
      const users = await User.findAll({ where: { companyId } });
      return users;
    } catch (error) {
      logger.error(`Error fetching users by company: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }

  /**
   * ユーザー情報を更新する
   */
  async updateUser(id: string, params: UserUpdateParams): Promise<User> {
    try {
      logger.info(`Updating user with ID: ${id}`);
      
      const user = await User.findByPk(id);
      if (!user) {
        logger.warn(`Update failed: User not found with ID: ${id}`);
        throw new Error('User not found');
      }
      
      // 更新データの準備
      const updateData: Partial<User> = {};
      
      if (params.name) updateData.name = params.name;
      if (params.email) {
        // メールアドレス変更時は重複チェック
        if (params.email !== user.email) {
          const existingUser = await User.findOne({ where: { email: params.email } });
          if (existingUser) {
            logger.warn(`Update failed: Email ${params.email} already in use`);
            throw new Error('Email already in use');
          }
          updateData.email = params.email;
        }
      }
      if (params.password) {
        updateData.password = await hash(params.password, SALT_ROUNDS);
      }
      if (params.role) updateData.role = params.role;
      
      // ユーザー更新
      await user.update(updateData);
      
      logger.info(`User ${id} updated successfully`);
      return user;
    } catch (error) {
      logger.error(`Error updating user: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }

  /**
   * ユーザーを削除する
   */
  async deleteUser(id: string): Promise<boolean> {
    try {
      logger.info(`Deleting user with ID: ${id}`);
      
      const user = await User.findByPk(id);
      if (!user) {
        logger.warn(`Deletion failed: User not found with ID: ${id}`);
        throw new Error('User not found');
      }
      
      await user.destroy();
      
      logger.info(`User ${id} deleted successfully`);
      return true;
    } catch (error) {
      logger.error(`Error deleting user: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }

  /**
   * JWTトークンを検証する
   */
  verifyToken(token: string): { userId: string; email: string; role: UserRole } {
    try {
      logger.info('Verifying JWT token');
      
      const decoded = verify(token, JWT_SECRET) as { userId: string; email: string; role: UserRole };
      return decoded;
    } catch (error) {
      logger.error(`Token verification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw new Error('Invalid token');
    }
  }

  /**
   * ユーザーの権限を検証する
   */
  checkPermission(userRole: UserRole, requiredRole: UserRole): boolean {
    // 単純な権限チェック。必要に応じて複雑なロジックに拡張可能
    const roleHierarchy = {
      [UserRole.ADMIN]: 3,
      [UserRole.MANAGER]: 2,
      [UserRole.USER]: 1
    };
    
    return roleHierarchy[userRole] >= roleHierarchy[requiredRole];
  }
}

export default new UserService();