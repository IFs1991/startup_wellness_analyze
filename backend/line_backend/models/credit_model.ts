import { Model, DataTypes, Association, ValidationError } from 'sequelize';
import { sequelize } from '../../database/connection';
import { User } from './user_model';
import { CreditTransaction } from './credit_transaction_model';

export interface CreditAttributes {
  id: string;
  vcId: string;
  portfolioCompanyId: string;
  totalCredits: number;
  usedCredits: number;
  remainingCredits: number;
  expirationDate: Date;
  updatedAt: Date;
  createdAt: Date;
}

export interface CreditCreationAttributes {
  vcId: string;
  portfolioCompanyId: string;
  totalCredits: number;
  expirationDate: Date;
}

export class Credit extends Model<CreditAttributes, CreditCreationAttributes> implements CreditAttributes {
  public id!: string;
  public vcId!: string;
  public portfolioCompanyId!: string;
  public totalCredits!: number;
  public usedCredits!: number;
  public remainingCredits!: number;
  public expirationDate!: Date;
  public readonly createdAt!: Date;
  public readonly updatedAt!: Date;

  // リレーションシップの定義
  public static associations: {
    vc: Association<Credit, User>;
    portfolioCompany: Association<Credit, User>;
    transactions: Association<Credit, CreditTransaction>;
  };

  // クレジットを使用するメソッド
  public async useCredits(amount: number): Promise<void> {
    if (amount <= 0) {
      throw new ValidationError('クレジット使用量は正の数である必要があります');
    }

    if (amount > this.remainingCredits) {
      throw new ValidationError('使用可能なクレジットが不足しています');
    }

    if (new Date() > this.expirationDate) {
      throw new ValidationError('クレジットの有効期限が切れています');
    }

    this.usedCredits += amount;
    this.remainingCredits -= amount;
    await this.save();
  }

  // クレジットを追加するメソッド
  public async addCredits(amount: number): Promise<void> {
    if (amount <= 0) {
      throw new ValidationError('追加クレジット量は正の数である必要があります');
    }

    this.totalCredits += amount;
    this.remainingCredits += amount;
    await this.save();
  }

  // クレジットの有効期限を更新するメソッド
  public async updateExpirationDate(newDate: Date): Promise<void> {
    if (newDate <= new Date()) {
      throw new ValidationError('有効期限は現在より未来の日付である必要があります');
    }

    this.expirationDate = newDate;
    await this.save();
  }

  // クレジットが有効かどうかを確認するメソッド
  public isValid(): boolean {
    return this.remainingCredits > 0 && new Date() <= this.expirationDate;
  }
}

Credit.init(
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    vcId: {
      type: DataTypes.UUID,
      allowNull: false,
      references: {
        model: 'users',
        key: 'id',
      },
      validate: {
        notEmpty: true,
      },
    },
    portfolioCompanyId: {
      type: DataTypes.UUID,
      allowNull: false,
      references: {
        model: 'users',
        key: 'id',
      },
      validate: {
        notEmpty: true,
      },
    },
    totalCredits: {
      type: DataTypes.INTEGER,
      allowNull: false,
      validate: {
        min: 0,
      },
    },
    usedCredits: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: 0,
      validate: {
        min: 0,
      },
    },
    remainingCredits: {
      type: DataTypes.INTEGER,
      allowNull: false,
      validate: {
        min: 0,
      },
    },
    expirationDate: {
      type: DataTypes.DATE,
      allowNull: false,
      validate: {
        isDate: true,
        isAfterToday(value: Date) {
          if (new Date(value) <= new Date()) {
            throw new Error('有効期限は現在より未来の日付である必要があります');
          }
        },
      },
    },
    createdAt: {
      type: DataTypes.DATE,
      allowNull: false,
    },
    updatedAt: {
      type: DataTypes.DATE,
      allowNull: false,
    },
  },
  {
    sequelize,
    tableName: 'credit_management',
    modelName: 'Credit',
    hooks: {
      beforeValidate: (credit: Credit) => {
        if (credit.isNewRecord) {
          credit.remainingCredits = credit.totalCredits;
          credit.usedCredits = 0;
        }
      },
    },
  }
);

// リレーションシップの設定
export const initCreditAssociations = (): void => {
  Credit.belongsTo(User, { as: 'vc', foreignKey: 'vcId' });
  Credit.belongsTo(User, { as: 'portfolioCompany', foreignKey: 'portfolioCompanyId' });
  Credit.hasMany(CreditTransaction, { as: 'transactions', foreignKey: 'creditId' });
};

export default Credit;