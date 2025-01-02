import { Company } from '@/types/company';
import { CompanyCard } from './CompanyCard';

interface CompanyListProps {
  companies: Company[];
  onCompanyClick: (company: Company) => void;
}

export function CompanyList({ companies, onCompanyClick }: CompanyListProps) {
  return (
    <div className="grid gap-4">
      {companies.map((company) => (
        <CompanyCard
          key={company.id}
          company={company}
          onClick={() => onCompanyClick(company)}
        />
      ))}
    </div>
  );
}