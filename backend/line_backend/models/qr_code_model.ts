import { Entity, Column, PrimaryGeneratedColumn, ManyToOne, JoinColumn, CreateDateColumn, UpdateDateColumn, BeforeInsert, BeforeUpdate } from 'typeorm';
import { IsEnum, IsDate, IsUUID, IsNotEmpty, IsInt, Min, IsOptional } from 'class-validator';
import { User } from './user_model';
import { Company } from './company_model';

export enum QRCodeType {
  SINGLE_USE = 'single_use',
  PERSISTENT = 'persistent'
}

export enum QRCodeStatus {
  ACTIVE = 'active',
  USED = 'used',
  EXPIRED = 'expired',
  CANCELLED = 'cancelled'
}

@Entity('qr_codes')
export class QRCode {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid', nullable: false })
  @IsUUID()
  @IsNotEmpty({ message: 'ユーザーIDは必須です' })
  userId: string;

  @Column({ type: 'uuid', nullable: false })
  @IsUUID()
  @IsNotEmpty({ message: '企業IDは必須です' })
  companyId: string;

  @Column({
    type: 'enum',
    enum: QRCodeType,
    default: QRCodeType.SINGLE_USE
  })
  @IsEnum(QRCodeType, { message: 'QRコードの種類は単回使用または永続のいずれかである必要があります' })
  type: QRCodeType;

  @Column({ type: 'int', nullable: false })
  @IsInt({ message: 'クレジット値は整数である必要があります' })
  @Min(1, { message: 'クレジット値は1以上である必要があります' })
  creditValue: number;

  @CreateDateColumn()
  createdAt: Date;

  @Column({ type: 'timestamp', nullable: false })
  @IsDate({ message: '有効期限は日付形式である必要があります' })
  expiresAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  @IsOptional()
  @IsDate({ message: '使用日時は日付形式である必要があります' })
  usedAt: Date | null;

  @Column({ type: 'uuid', nullable: true })
  @IsOptional()
  @IsUUID()
  staffId: string | null;

  @Column({
    type: 'enum',
    enum: QRCodeStatus,
    default: QRCodeStatus.ACTIVE
  })
  @IsEnum(QRCodeStatus, { message: 'ステータスは有効、使用済み、期限切れ、キャンセル済みのいずれかである必要があります' })
  status: QRCodeStatus;

  @UpdateDateColumn()
  updatedAt: Date;

  @ManyToOne(() => User, user => user.qrCodes)
  @JoinColumn({ name: 'userId' })
  user: User;

  @ManyToOne(() => Company, company => company.qrCodes)
  @JoinColumn({ name: 'companyId' })
  company: Company;

  @BeforeInsert()
  @BeforeUpdate()
  updateStatusBasedOnDates() {
    const now = new Date();
    
    // 有効期限が過ぎている場合
    if (this.expiresAt < now && this.status === QRCodeStatus.ACTIVE) {
      this.status = QRCodeStatus.EXPIRED;
    }
    
    // 使用済みの場合
    if (this.usedAt && this.status === QRCodeStatus.ACTIVE) {
      this.status = QRCodeStatus.USED;
    }
  }

  // QRコードの使用を記録するメソッド
  markAsUsed(staffId: string): boolean {
    if (this.status !== QRCodeStatus.ACTIVE) {
      return false;
    }

    this.usedAt = new Date();
    this.staffId = staffId;
    this.status = QRCodeStatus.USED;
    return true;
  }

  // QRコードをキャンセルするメソッド
  cancel(): boolean {
    if (this.status !== QRCodeStatus.ACTIVE) {
      return false;
    }
    
    this.status = QRCodeStatus.CANCELLED;
    return true;
  }

  // QRコードが有効かどうかを確認するメソッド
  isValid(): boolean {
    return this.status === QRCodeStatus.ACTIVE && new Date() < this.expiresAt;
  }
}