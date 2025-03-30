import { Schema, model, Document, Model } from 'mongoose';
import bcrypt from 'bcryptjs';
import validator from 'validator';

// ユーザー権限の列挙型
export enum UserRole {
  ADMIN = 'admin',
  MANAGER = 'manager',
  STAFF = 'staff',
  USER = 'user'
}

// ユーザーインターフェース
export interface IUser extends Document {
  lineUserId?: string;
  companyId: string;
  fullName: string;
  email: string;
  password: string;
  role: UserRole;
  createdAt: Date;
  updatedAt: Date;
  comparePassword(candidatePassword: string): Promise<boolean>;
}

// ユーザーモデルインターフェース
export interface IUserModel extends Model<IUser> {
  findByEmail(email: string): Promise<IUser | null>;
  findByLineUserId(lineUserId: string): Promise<IUser | null>;
}

// バリデーションルール
const userSchema = new Schema<IUser>(
  {
    lineUserId: {
      type: String,
      unique: true,
      sparse: true,
      index: true
    },
    companyId: {
      type: String,
      required: [true, '企業IDは必須です'],
      index: true
    },
    fullName: {
      type: String,
      required: [true, '氏名は必須です'],
      trim: true,
      maxlength: [50, '氏名は50文字以内で入力してください']
    },
    email: {
      type: String,
      required: [true, 'メールアドレスは必須です'],
      unique: true,
      lowercase: true,
      validate: [validator.isEmail, '有効なメールアドレスを入力してください'],
      index: true
    },
    password: {
      type: String,
      required: [true, 'パスワードは必須です'],
      minlength: [8, 'パスワードは8文字以上で入力してください'],
      select: false // デフォルトでクエリに含めない
    },
    role: {
      type: String,
      enum: Object.values(UserRole),
      default: UserRole.USER
    },
    createdAt: {
      type: Date,
      default: Date.now
    },
    updatedAt: {
      type: Date,
      default: Date.now
    }
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
  }
);

// インデックスの設定
userSchema.index({ companyId: 1, email: 1 }, { unique: true });

// 仮想フィールド: 関連するクレジット情報
userSchema.virtual('credits', {
  ref: 'Credit',
  localField: '_id',
  foreignField: 'userId'
});

// 仮想フィールド: 関連する予約情報
userSchema.virtual('reservations', {
  ref: 'Reservation',
  localField: '_id',
  foreignField: 'userId'
});

// パスワードハッシュ化のミドルウェア
userSchema.pre('save', async function(next) {
  // パスワードが変更されていない場合はスキップ
  if (!this.isModified('password')) return next();

  try {
    // パスワードのハッシュ化
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (error: any) {
    next(error);
  }
});

// パスワード比較メソッド
userSchema.methods.comparePassword = async function(candidatePassword: string): Promise<boolean> {
  return bcrypt.compare(candidatePassword, this.password);
};

// 静的メソッド: メールアドレスでユーザーを検索
userSchema.statics.findByEmail = async function(email: string): Promise<IUser | null> {
  return this.findOne({ email });
};

// 静的メソッド: LINE IDでユーザーを検索
userSchema.statics.findByLineUserId = async function(lineUserId: string): Promise<IUser | null> {
  return this.findOne({ lineUserId });
};

// モデルの作成と出力
const UserModel = model<IUser, IUserModel>('User', userSchema);

export default UserModel;